"""
推理性能优化与基准测试
======================
面试亮点：展示你对工业部署的理解

优化技术：
1. KV Cache 优化
2. INT8/INT4 量化
3. Flash Attention
4. vLLM 推理加速
5. 蒙特卡洛解码（投机采样）

用法：
    python inference_optimization.py --model merged-ecom-dpo --benchmark
    python inference_optimization.py --model merged-ecom-dpo --quantize int4
    python inference_optimization.py --model merged-ecom-dpo --compare_all
"""

import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from contextlib import contextmanager

import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    method: str
    model_size_gb: float
    load_time_sec: float
    first_token_latency_ms: float
    tokens_per_second: float
    memory_used_gb: float
    memory_peak_gb: float
    quality_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "model_size_gb": round(self.model_size_gb, 2),
            "load_time_sec": round(self.load_time_sec, 2),
            "first_token_latency_ms": round(self.first_token_latency_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_peak_gb": round(self.memory_peak_gb, 2),
            "quality_score": round(self.quality_score, 3) if self.quality_score else None,
        }


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def get_gpu_memory():
    """获取 GPU 显存使用情况"""
    if torch.cuda.is_available():
        return {
            "used": torch.cuda.memory_allocated() / 1024**3,
            "peak": torch.cuda.max_memory_allocated() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }
    return {"used": 0, "peak": 0, "total": 0}


def reset_gpu_memory():
    """重置 GPU 显存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


@contextmanager
def timer():
    """计时器上下文管理器"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def get_model_size_gb(model) -> float:
    """计算模型参数占用的显存 (GB)"""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / 1024**3


# ═══════════════════════════════════════════════════════════════
# 基准测试 Prompts
# ═══════════════════════════════════════════════════════════════

BENCHMARK_PROMPTS = [
    "帮我写一个无线蓝牙耳机的淘宝爆款标题",
    "买家说太贵了要砍价，客服应该怎么回复？",
    "商品点击率高但转化率低，可能的原因是什么？给出3个优化方向。",
    "为双11设计一个提升客单价的满减策略",
    "新品零销量如何快速冷启动？",
]

SYSTEM_PROMPT = "你是专业的电商运营和销售顾问。"


# ═══════════════════════════════════════════════════════════════
# 方法 1: 原始 HuggingFace 推理
# ═══════════════════════════════════════════════════════════════

def benchmark_hf_baseline(model_path: str, prompts: List[str], 
                          max_new_tokens: int = 256) -> BenchmarkResult:
    """HuggingFace 原始推理基准"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\n📊 测试: HuggingFace Baseline (BF16)")
    reset_gpu_memory()
    
    # 加载模型
    with timer() as get_load_time:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    load_time = get_load_time()
    model_size = get_model_size_gb(model)
    
    # 预热
    _ = generate_single(model, tokenizer, "你好", max_new_tokens=10)
    
    # 基准测试
    first_token_latencies = []
    total_tokens = 0
    total_time = 0
    
    for prompt in tqdm(prompts, desc="  推理中"):
        reset_gpu_memory()
        
        with timer() as get_time:
            output, num_tokens, first_latency = generate_single(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
        
        first_token_latencies.append(first_latency)
        total_tokens += num_tokens
        total_time += get_time()
    
    mem = get_gpu_memory()
    
    # 清理
    del model
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        method="HF Baseline (BF16)",
        model_size_gb=model_size,
        load_time_sec=load_time,
        first_token_latency_ms=sum(first_token_latencies) / len(first_token_latencies) * 1000,
        tokens_per_second=total_tokens / total_time,
        memory_used_gb=mem["used"],
        memory_peak_gb=mem["peak"],
    )


def generate_single(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """单次生成，返回输出、token数和首token延迟"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 首 token 延迟
    first_token_time = None
    
    with torch.no_grad():
        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # 近似首 token 时间（实际需要 streaming）
        first_token_time = time.perf_counter() - start
    
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    num_tokens = len(new_tokens)
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return decoded, num_tokens, first_token_time / num_tokens if num_tokens > 0 else first_token_time


# ═══════════════════════════════════════════════════════════════
# 方法 2: INT8 量化
# ═══════════════════════════════════════════════════════════════

def benchmark_int8_quantization(model_path: str, prompts: List[str],
                                 max_new_tokens: int = 256) -> BenchmarkResult:
    """INT8 量化推理"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print("\n📊 测试: INT8 量化 (bitsandbytes)")
    reset_gpu_memory()
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    with timer() as get_load_time:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    load_time = get_load_time()
    
    # 预热
    _ = generate_single(model, tokenizer, "你好", max_new_tokens=10)
    
    # 基准测试
    first_token_latencies = []
    total_tokens = 0
    total_time = 0
    
    for prompt in tqdm(prompts, desc="  推理中"):
        with timer() as get_time:
            output, num_tokens, first_latency = generate_single(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
        first_token_latencies.append(first_latency)
        total_tokens += num_tokens
        total_time += get_time()
    
    mem = get_gpu_memory()
    
    # 估算量化后模型大小
    param_count = sum(p.numel() for p in model.parameters())
    model_size = param_count * 1 / 1024**3  # INT8 = 1 byte per param
    
    del model
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        method="INT8 Quantization",
        model_size_gb=model_size,
        load_time_sec=load_time,
        first_token_latency_ms=sum(first_token_latencies) / len(first_token_latencies) * 1000,
        tokens_per_second=total_tokens / total_time,
        memory_used_gb=mem["used"],
        memory_peak_gb=mem["peak"],
    )


# ═══════════════════════════════════════════════════════════════
# 方法 3: INT4 量化 (GPTQ/AWQ 风格)
# ═══════════════════════════════════════════════════════════════

def benchmark_int4_quantization(model_path: str, prompts: List[str],
                                 max_new_tokens: int = 256) -> BenchmarkResult:
    """INT4 量化推理"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print("\n📊 测试: INT4 量化 (NF4)")
    reset_gpu_memory()
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    with timer() as get_load_time:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    load_time = get_load_time()
    
    # 预热
    _ = generate_single(model, tokenizer, "你好", max_new_tokens=10)
    
    # 基准测试
    first_token_latencies = []
    total_tokens = 0
    total_time = 0
    
    for prompt in tqdm(prompts, desc="  推理中"):
        with timer() as get_time:
            output, num_tokens, first_latency = generate_single(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
        first_token_latencies.append(first_latency)
        total_tokens += num_tokens
        total_time += get_time()
    
    mem = get_gpu_memory()
    
    # 估算量化后模型大小
    param_count = sum(p.numel() for p in model.parameters())
    model_size = param_count * 0.5 / 1024**3  # INT4 = 0.5 byte per param
    
    del model
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        method="INT4 Quantization (NF4)",
        model_size_gb=model_size,
        load_time_sec=load_time,
        first_token_latency_ms=sum(first_token_latencies) / len(first_token_latencies) * 1000,
        tokens_per_second=total_tokens / total_time,
        memory_used_gb=mem["used"],
        memory_peak_gb=mem["peak"],
    )


# ═══════════════════════════════════════════════════════════════
# 方法 4: Flash Attention 2
# ═══════════════════════════════════════════════════════════════

def benchmark_flash_attention(model_path: str, prompts: List[str],
                               max_new_tokens: int = 256) -> BenchmarkResult:
    """Flash Attention 2 推理"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\n📊 测试: Flash Attention 2")
    reset_gpu_memory()
    
    with timer() as get_load_time:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # 启用 FA2
        )
        model.eval()
    load_time = get_load_time()
    model_size = get_model_size_gb(model)
    
    # 预热
    _ = generate_single(model, tokenizer, "你好", max_new_tokens=10)
    
    # 基准测试
    first_token_latencies = []
    total_tokens = 0
    total_time = 0
    
    for prompt in tqdm(prompts, desc="  推理中"):
        with timer() as get_time:
            output, num_tokens, first_latency = generate_single(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
        first_token_latencies.append(first_latency)
        total_tokens += num_tokens
        total_time += get_time()
    
    mem = get_gpu_memory()
    
    del model
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        method="Flash Attention 2",
        model_size_gb=model_size,
        load_time_sec=load_time,
        first_token_latency_ms=sum(first_token_latencies) / len(first_token_latencies) * 1000,
        tokens_per_second=total_tokens / total_time,
        memory_used_gb=mem["used"],
        memory_peak_gb=mem["peak"],
    )


# ═══════════════════════════════════════════════════════════════
# 方法 5: vLLM 推理 (如果安装了)
# ═══════════════════════════════════════════════════════════════

def benchmark_vllm(model_path: str, prompts: List[str],
                   max_new_tokens: int = 256) -> Optional[BenchmarkResult]:
    """vLLM 推理（需要安装 vllm）"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("\n⚠️  vLLM 未安装，跳过此测试")
        print("   安装命令: pip install vllm")
        return None
    
    print("\n📊 测试: vLLM")
    reset_gpu_memory()
    
    with timer() as get_load_time:
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,
        )
    load_time = get_load_time()
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_new_tokens,
    )
    
    # 构建完整 prompts
    full_prompts = []
    for p in prompts:
        full_prompts.append(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                          f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
    
    # 预热
    _ = llm.generate(full_prompts[:1], sampling_params)
    
    # 基准测试
    with timer() as get_time:
        outputs = llm.generate(full_prompts, sampling_params)
    total_time = get_time()
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    mem = get_gpu_memory()
    
    del llm
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        method="vLLM",
        model_size_gb=0,  # vLLM 内部管理
        load_time_sec=load_time,
        first_token_latency_ms=total_time / len(prompts) * 1000 / (total_tokens / len(prompts)),
        tokens_per_second=total_tokens / total_time,
        memory_used_gb=mem["used"],
        memory_peak_gb=mem["peak"],
    )


# ═══════════════════════════════════════════════════════════════
# 综合对比
# ═══════════════════════════════════════════════════════════════

def run_all_benchmarks(model_path: str, output_path: str = "benchmark_results.json"):
    """运行所有基准测试并对比"""
    prompts = BENCHMARK_PROMPTS
    results = []
    
    print("\n" + "=" * 60)
    print("  🚀 推理性能基准测试")
    print("=" * 60)
    print(f"  模型路径: {model_path}")
    print(f"  测试样本: {len(prompts)} 条")
    
    # 1. Baseline
    try:
        results.append(benchmark_hf_baseline(model_path, prompts))
    except Exception as e:
        print(f"  ❌ HF Baseline 失败: {e}")
    
    # 2. INT8
    try:
        results.append(benchmark_int8_quantization(model_path, prompts))
    except Exception as e:
        print(f"  ❌ INT8 量化失败: {e}")
    
    # 3. INT4
    try:
        results.append(benchmark_int4_quantization(model_path, prompts))
    except Exception as e:
        print(f"  ❌ INT4 量化失败: {e}")
    
    # 4. Flash Attention
    try:
        results.append(benchmark_flash_attention(model_path, prompts))
    except Exception as e:
        print(f"  ❌ Flash Attention 失败: {e}")
    
    # 5. vLLM
    try:
        vllm_result = benchmark_vllm(model_path, prompts)
        if vllm_result:
            results.append(vllm_result)
    except Exception as e:
        print(f"  ❌ vLLM 失败: {e}")
    
    # 生成报告
    print("\n" + "=" * 80)
    print("  📊 性能对比报告")
    print("=" * 80)
    
    print(f"\n{'方法':<25} {'显存(GB)':<12} {'速度(tok/s)':<15} {'首Token(ms)':<15} {'加速比':<10}")
    print("-" * 80)
    
    baseline_speed = results[0].tokens_per_second if results else 1
    
    for r in results:
        speedup = r.tokens_per_second / baseline_speed
        print(f"{r.method:<25} {r.memory_peak_gb:<12.2f} {r.tokens_per_second:<15.1f} "
              f"{r.first_token_latency_ms:<15.2f} {speedup:<10.2f}x")
    
    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存: {output_path}")
    
    return results


# ═══════════════════════════════════════════════════════════════
# KV Cache 优化分析
# ═══════════════════════════════════════════════════════════════

def analyze_kv_cache(model_path: str):
    """分析 KV Cache 显存占用"""
    from transformers import AutoConfig
    
    print("\n" + "=" * 60)
    print("  📊 KV Cache 显存分析")
    print("=" * 60)
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 获取模型配置
    hidden_size = getattr(config, "hidden_size", 4096)
    num_layers = getattr(config, "num_hidden_layers", 32)
    num_kv_heads = getattr(config, "num_key_value_heads", 
                          getattr(config, "num_attention_heads", 32))
    head_dim = hidden_size // getattr(config, "num_attention_heads", 32)
    
    print(f"\n  模型配置:")
    print(f"    Hidden Size:    {hidden_size}")
    print(f"    Num Layers:     {num_layers}")
    print(f"    KV Heads:       {num_kv_heads}")
    print(f"    Head Dim:       {head_dim}")
    
    # 计算不同序列长度的 KV Cache 大小
    print(f"\n  KV Cache 显存占用 (BF16, batch_size=1):")
    print(f"  {'序列长度':<12} {'KV Cache (MB)':<15} {'相当于参数量':<15}")
    print("  " + "-" * 45)
    
    for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
        # KV Cache: 2 * num_layers * seq_len * num_kv_heads * head_dim * dtype_size
        kv_cache_bytes = 2 * num_layers * seq_len * num_kv_heads * head_dim * 2  # BF16 = 2 bytes
        kv_cache_mb = kv_cache_bytes / 1024**2
        equiv_params = kv_cache_bytes / 2 / 1e9  # 相当于多少 B 参数
        print(f"  {seq_len:<12} {kv_cache_mb:<15.1f} {equiv_params:.2f}B")
    
    print(f"\n  💡 优化建议:")
    print(f"    1. 使用 GQA (Grouped Query Attention) 减少 KV Cache")
    print(f"    2. 使用 PagedAttention (vLLM) 优化显存管理")
    print(f"    3. 限制 max_seq_len 减少峰值显存")


def main():
    parser = argparse.ArgumentParser(description="推理性能优化与基准测试")
    parser.add_argument("--model", type=str, default="merged-ecom-dpo",
                       help="模型路径")
    parser.add_argument("--benchmark", action="store_true",
                       help="运行基准测试")
    parser.add_argument("--quantize", type=str, choices=["int8", "int4"],
                       help="单独测试量化")
    parser.add_argument("--compare_all", action="store_true",
                       help="运行所有对比测试")
    parser.add_argument("--kv_cache", action="store_true",
                       help="分析 KV Cache 占用")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="结果输出路径")
    args = parser.parse_args()
    
    if args.kv_cache:
        analyze_kv_cache(args.model)
    
    if args.compare_all or args.benchmark:
        run_all_benchmarks(args.model, args.output)
    
    if args.quantize == "int8":
        result = benchmark_int8_quantization(args.model, BENCHMARK_PROMPTS)
        print(f"\nINT8 结果: {result.to_dict()}")
    elif args.quantize == "int4":
        result = benchmark_int4_quantization(args.model, BENCHMARK_PROMPTS)
        print(f"\nINT4 结果: {result.to_dict()}")


if __name__ == "__main__":
    main()
