# 🛒 金融领域大模型训练与优化项目

> **面试级项目**：完整展示 LLM 算法岗核心能力

## 📋 项目概览

基于 Qwen2.5 的金融领域大模型，覆盖 **PT → SFT → DPO** 全流程训练，并实现**消融实验、推理优化、自动评估、数据质量分析**四大工业级能力。

### 项目亮点

| 模块 | 技术点 | 面试考察点 |
|------|--------|-----------|
| 三阶段训练 | PT + SFT + DPO | 训练流程理解、超参调优 |
| 消融实验 | β/lr/rank 影响分析 | 实验设计能力、原理理解 |
| 推理优化 | 量化/FlashAttn/vLLM | 工程部署能力 |
| 自动评估 | GPT评分 + 规则指标 | 效果衡量能力 |
| 数据分析 | 去重/过滤/分层 | 数据思维 |

## 🗂️ 项目结构

```
ecommerce_llm_project/
├── datasets/
│   ├── evol_diversity.py          
│   ├── generate_financial_dataset_v3.py
│   ├── template.py
│   └── multiturn_dialogue.py         
│
├── experiments/
│   └── dpo_ablation.py    # ⭐ DPO 超参消融实验
│
├── optimization/
│   └── inference_optimization.py  # ⭐ 推理性能优化
│
├── evaluation/
│   └── auto_evaluation.py # ⭐ 自动化评估体系
│
├── analysis/
│   └── data_quality_pipeline.py  # ⭐ 数据质量分析
│
├── configs/               # 训练配置
├── scripts/               # 辅助脚本
├── generate_ecommerce_dataset.py  # 数据生成器
└── ecommerce_dpo_training_pipeline.ipynb  # 训练 Pipeline
```

## 🚀 快速开始

### 环境配置

```bash
# 基础依赖
pip install transformers==4.46.0 peft==0.13.0 trl==0.12.0 \
            datasets accelerate bitsandbytes sentencepiece

# 推理优化（可选）
pip install vllm flash-attn

# 评估（可选）
pip install openai
```

### 1. 数据生成

```bash
# 确保 Ollama 运行中
ollama pull qwen2.5:7b

# 生成三阶段数据
python generate_ecommerce_dataset.py --pt 20 --sft 100 --dpo 50
```

### 2. 训练流程

```bash
# 参考 Notebook 中的训练命令
# Stage 1: Continue PreTraining (可选)
# Stage 2: SFT 微调
# Stage 3: DPO 对齐
```

### 3. 消融实验

```bash
# β 参数消融
python experiments/dpo_ablation.py --exp beta --base_model merged-ecom-sft

# 学习率消融
python experiments/dpo_ablation.py --exp lr --base_model merged-ecom-sft

# 全部实验
python experiments/dpo_ablation.py --exp all --base_model merged-ecom-sft
```

### 4. 推理优化

```bash
# 运行所有基准测试
python optimization/inference_optimization.py --model merged-ecom-dpo --compare_all

# 单独测试量化
python optimization/inference_optimization.py --model merged-ecom-dpo --quantize int4

# KV Cache 分析
python optimization/inference_optimization.py --model merged-ecom-dpo --kv_cache
```

### 5. 自动评估

```bash
# 规则评估（不需要 API）
python evaluation/auto_evaluation.py --model_a merged-ecom-sft --model_b merged-ecom-dpo --method rule

# GPT-4 评估（需要 OPENAI_API_KEY）
export OPENAI_API_KEY=sk-xxx
python evaluation/auto_evaluation.py --model_a merged-ecom-sft --model_b merged-ecom-dpo --method gpt
```

### 6. 数据质量分析

```bash
# 完整分析
python analysis/data_quality_analysis.py --data ./data/finetune/ecommerce_sft.jsonl --analyze

# 去重
python analysis/data_quality_analysis.py --data ./data/finetune/ecommerce_sft.jsonl --dedup --output cleaned.jsonl
```

---

## 📊 实验结果（示例）

### DPO 消融实验

| 实验 | β | Learning Rate | Reward Margin | 结论 |
|------|---|---------------|---------------|------|
| beta_0.05 | 0.05 | 5e-7 | 0.82 | 最稳定 |
| beta_0.1 | 0.1 | 5e-7 | 1.15 | 默认推荐 |
| beta_0.2 | 0.2 | 5e-7 | 1.43 | 效果好但风险高 |
| lr_1e-6 | 0.1 | 1e-6 | 0.95 | 收敛快 |
| lr_1e-5 | 0.1 | 1e-5 | -0.2 | 训练崩溃 |

**关键发现**：
- β 越大，偏好学习越激进，但容易过拟合
- DPO 对学习率极其敏感，>1e-5 基本崩溃
- Rank=8 在小数据集上足够，数据量大时可提升到 16

### 推理优化对比

| 方法 | 显存 (GB) | 速度 (tok/s) | 加速比 |
|------|-----------|--------------|--------|
| HF Baseline | 14.2 | 45 | 1.0x |
| INT8 量化 | 8.5 | 52 | 1.15x |
| INT4 量化 | 5.8 | 48 | 1.07x |
| Flash Attention 2 | 12.1 | 68 | 1.51x |
| vLLM | 11.5 | 120 | 2.67x |

### 模型效果对比 (SFT vs DPO)

| 类别 | SFT 胜率 | DPO 胜率 | 平局 |
|------|----------|----------|------|
| 文案生成 | 20% | 60% | 20% |
| 客服话术 | 30% | 50% | 20% |
| 差评处理 | 25% | 55% | 20% |
| 运营策略 | 20% | 65% | 15% |
| **总体** | **24%** | **57%** | **19%** |

---

## 🎯 面试准备指南

### 必答问题清单

#### 1. DPO 原理
**Q: 请解释 DPO 的核心思想？与 PPO 有什么区别？**

A: DPO 将 RLHF 中的 reward model 和 RL 训练合并为一步：
- 核心公式：`L_DPO = -log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))`
- 直接用偏好数据优化策略，无需训练单独的 reward model
- β 控制与参考模型的 KL 散度惩罚

vs PPO:
- PPO 需要 4 个模型（policy, reference, reward, value）
- DPO 只需要 2 个模型（policy, reference）
- DPO 更稳定，但 PPO 理论上限更高

#### 2. 为什么做消融实验？
**Q: 你做的消融实验有什么发现？**

A: 
- **β 消融**：β=0.1 是较好的默认值；β 过小学不到偏好，过大容易过拟合
- **LR 消融**：DPO 对学习率极其敏感，5e-7 ~ 1e-6 是安全区间
- **Rank 消融**：数据量小时 rank=8 足够，数据量大可用 16-32

#### 3. 如何评估模型效果？
**Q: 你怎么知道 DPO 比 SFT 好？**

A:
- **自动评估**：GPT-4 pairwise comparison，57% 胜率
- **规则评估**：关键词覆盖、结构完整性、长度适中
- **人工评估**：盲评模板，避免位置偏见

#### 4. 推理优化怎么做？
**Q: 部署时如何优化推理速度？**

A:
- **量化**：INT8 显存降 40%，速度略提升；INT4 显存降 60%，精度有损失
- **Flash Attention**：速度提升 50%，显存降 15%
- **vLLM**：PagedAttention + Continuous Batching，速度提升 2.6x
- **KV Cache**：长序列时是显存瓶颈，GQA 可有效减少

#### 5. 数据质量怎么保证？
**Q: 工业界数据通常有什么问题？怎么处理？**

A:
- **重复**：n-gram / embedding 去重
- **低质量**：规则过滤 + 模型打分
- **分布不均**：难度分层，课程学习
- **标注噪声**：多人标注取交集，置信度过滤

### 技术深挖问题

**Q: LoRA 为什么有效？**
- 大模型微调时，权重更新矩阵是低秩的
- LoRA 用 A×B 分解，大幅减少可训练参数
- 训练时只更新 A 和 B，推理时合并回原权重

**Q: 为什么 DPO 要用 reference model？**
- 防止模型偏离原始分布太远
- β 控制 KL 散度惩罚强度
- 保持模型的通用能力

**Q: Reward Hacking 是什么？怎么避免？**
- 模型学会欺骗 reward signal 而非真正提升质量
- 例如：回答变长、重复高分关键词
- 解决：控制 β、多样化 reward、人工监控

---

## 📚 参考资料

- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [vLLM 论文](https://arxiv.org/abs/2309.06180)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)

---

## 📝 License

MIT License
