# -*- coding: utf-8 -*-
"""
多轮对话数据生成与训练模块
=====================================
基于 Self-Play + Evol-Instruct 的多轮对话数据合成方案

方案创新点：
  1. Self-Play 对话合成：LLM 分饰 User 与 Assistant 双角色自动构造多轮对话
     参考：SPIN (Self-Play Fine-Tuning, ICML 2024) 的自博弈思路
  2. 对话复杂度进化：基于 Evol-Instruct 对多轮对话进行深度追问、场景转换、反驳纠错
  3. 对话质量过滤：通过回合一致性、信息密度、角色一致性进行多维度筛选
  4. 与 LLaMA-Factory 深度集成，支持 sharegpt 多轮对话训练格式

Usage:
  python multiturn_dialogue.py \
      --model qwen2.5:7b \
      --num_dialogues 200 \
      --max_turns 6 \
      --output_dir ./data/multiturn
"""

import os
import json
import random
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# 电商多轮对话场景定义
# ══════════════════════════════════════════════════════════════════════

MULTITURN_SCENARIOS = [
    {
        "category": "售前咨询",
        "description": "买家对商品感兴趣，逐步深入询问细节",
        "example_flow": "商品基本信息 → 规格对比 → 使用场景 → 售后保障 → 下单决策",
        "products": ["无线蓝牙耳机", "扫地机器人", "电动牙刷", "空气炸锅", "投影仪"],
    },
    {
        "category": "售后处理",
        "description": "买家收货后遇到问题，需要多轮沟通解决",
        "example_flow": "问题描述 → 故障排查 → 解决方案 → 补偿协商 → 满意确认",
        "products": ["笔记本电脑", "智能手表", "按摩椅", "咖啡机", "破壁机"],
    },
    {
        "category": "运营策划",
        "description": "运营团队讨论活动方案，逐步细化策略",
        "example_flow": "目标设定 → 数据分析 → 方案制定 → 预算评估 → 执行排期",
        "products": ["女装连衣裙", "护肤套装", "零食大礼包", "儿童玩具", "运动鞋"],
    },
    {
        "category": "数据分析讨论",
        "description": "基于数据报表进行多轮分析和决策",
        "example_flow": "数据概览 → 异常定位 → 原因分析 → 优化建议 → 预期效果",
        "metrics": ["转化率", "客单价", "复购率", "ROI", "跳失率"],
    },
    {
        "category": "直播策划",
        "description": "直播前的多轮策划讨论",
        "example_flow": "选品讨论 → 话术设计 → 优惠方案 → 互动环节 → 复盘改进",
        "products": ["美妆礼盒", "家居收纳", "数码配件", "母婴用品", "健身器材"],
    },
    {
        "category": "砍价博弈",
        "description": "买家多轮砍价，客服灵活应对守住利润",
        "example_flow": "初次报价 → 买家砍价 → 价值说明 → 优惠方案 → 成交",
        "products": ["真皮沙发", "实木书桌", "品牌包包", "高端耳机", "智能门锁"],
    },
]

# ══════════════════════════════════════════════════════════════════════
# 对话进化策略（Evol-Instruct for Multi-turn）
# ══════════════════════════════════════════════════════════════════════

EVOLUTION_STRATEGIES = [
    {
        "name": "深度追问",
        "description": "在某个话题上持续深入，要求更具体的数据和方案",
        "prompt_suffix": "请在第{turn}轮让用户针对上一轮的某个具体细节进行深度追问，要求给出具体数据或案例。",
    },
    {
        "name": "场景切换",
        "description": "用户在对话中途提出新的关联需求",
        "prompt_suffix": "请在第{turn}轮让用户提出一个与当前话题相关但不同维度的新需求。",
    },
    {
        "name": "反驳纠错",
        "description": "用户对助手的建议提出质疑或反例",
        "prompt_suffix": "请在第{turn}轮让用户对助手的某个建议提出质疑或给出一个反例，助手需要合理回应。",
    },
    {
        "name": "信息补充",
        "description": "用户补充关键背景信息，改变问题约束",
        "prompt_suffix": "请在第{turn}轮让用户补充一个重要的约束条件（如预算限制、时间紧迫、特殊需求），助手据此调整方案。",
    },
]


# ══════════════════════════════════════════════════════════════════════
# 核心：Self-Play 多轮对话生成
# ══════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, model: str = "deepseek-chat", temperature: float = 0.8) -> str:
    from openai import OpenAI
    import os
    client = OpenAI(
        api_key="sk-655ad4c737eb40239b5fa7813aef62a7",
        base_url="https://api.deepseek.com",
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return ""


def generate_multiturn_dialogue(
    scenario: Dict,
    model: str = "qwen2.5:7b",
    max_turns: int = 6,
    min_turns: int = 3,
    evolution_prob: float = 0.4,
) -> Optional[Dict]:
    """
    使用 Self-Play 方式生成一段多轮对话。

    策略：
      1. 先让 LLM 生成对话开场（用户第一个问题）
      2. 交替扮演 User 和 Assistant，逐轮推进
      3. 随机插入进化策略，增加对话复杂度
      4. 控制对话长度在 min_turns ~ max_turns 之间
    """
    category = scenario["category"]
    description = scenario["description"]
    example_flow = scenario["example_flow"]

    # 随机选择产品/指标
    if "products" in scenario:
        context_item = random.choice(scenario["products"])
        context_type = "产品"
    elif "metrics" in scenario:
        context_item = random.choice(scenario["metrics"])
        context_type = "指标"
    else:
        context_item = "通用场景"
        context_type = "场景"

    num_turns = random.randint(min_turns, max_turns)

    # Step 1: 生成开场问题
    opening_prompt = f"""你现在扮演一位电商从业者（可能是店铺运营、客服主管或品牌负责人）。
请根据以下场景，提出一个真实、具体的初始问题：

场景类别：{category}
场景描述：{description}
对话流程参考：{example_flow}
涉及{context_type}：{context_item}

要求：
1. 问题必须具体，包含实际的业务背景
2. 不要说"我是一个运营"之类的自我介绍
3. 直接提出你的需求或问题
4. 语气自然，像真实的业务沟通

只输出用户的第一个问题，不要其他内容。"""

    first_question = call_llm(opening_prompt, model, temperature=0.9)
    if not first_question:
        return None

    # Step 2: 交替生成对话
    conversations = []
    dialogue_history = []

    conversations.append({"from": "human", "value": first_question})
    dialogue_history.append(f"用户：{first_question}")

    for turn_idx in range(num_turns):
        # 生成 Assistant 回复
        history_text = "\n".join(dialogue_history)

        assistant_prompt = f"""你是一位资深的电商运营专家和销售顾问，拥有 10 年以上的实战经验。
请根据对话历史，给出专业、详细、有实操价值的回复。

场景：{category} - {description}

对话历史：
{history_text}

要求：
1. 回复要专业且有深度，包含具体的数据、案例或步骤
2. 针对用户的具体问题给出可执行的方案
3. 如果用户的某个假设有误，温和地指出并给出正确建议
4. 语气专业但亲切
5. 回复长度适中（200-500字）

只输出助手的回复，不要角色标签。"""

        assistant_reply = call_llm(assistant_prompt, model, temperature=0.7)
        if not assistant_reply:
            break

        conversations.append({"from": "gpt", "value": assistant_reply})
        dialogue_history.append(f"助手：{assistant_reply}")

        # 如果还没到最后一轮，生成下一轮用户追问
        if turn_idx < num_turns - 1:
            # 决定是否使用进化策略
            if random.random() < evolution_prob:
                strategy = random.choice(EVOLUTION_STRATEGIES)
                evolution_hint = strategy["prompt_suffix"].format(turn=turn_idx + 2)
            else:
                evolution_hint = "请自然地进行追问或深化讨论。"

            user_prompt = f"""你继续扮演电商从业者，根据对话历史生成下一轮提问。

对话历史：
{history_text}
助手：{assistant_reply}

进化策略：{evolution_hint}

要求：
1. 追问要自然，衔接上文
2. 体现真实业务场景中的逐步深入
3. 可以对助手的某个观点提出疑问或要求更多细节
4. 不要重复之前已经讨论过的内容
5. 语气自然真实

只输出用户的下一个问题，不要角色标签。"""

            next_question = call_llm(user_prompt, model, temperature=0.9)
            if not next_question:
                break

            conversations.append({"from": "human", "value": next_question})
            dialogue_history.append(f"用户：{next_question}")

    # 验证对话质量
    if len(conversations) < min_turns * 2:
        return None

    # 确保最后一轮是 assistant 回复
    if conversations[-1]["from"] == "human":
        # 补一个结尾回复
        final_prompt = f"""你是电商运营专家。请给出一个简洁有力的总结性回复，结束这段对话。

对话历史：
{chr(10).join(dialogue_history)}

只输出最终回复。"""
        final_reply = call_llm(final_prompt, model, temperature=0.6)
        if final_reply:
            conversations.append({"from": "gpt", "value": final_reply})

    return {
        "conversations": conversations,
        "metadata": {
            "category": category,
            "context_item": context_item,
            "num_turns": len([c for c in conversations if c["from"] == "human"]),
            "generated_at": datetime.now().isoformat(),
        }
    }


# ══════════════════════════════════════════════════════════════════════
# 对话质量过滤
# ══════════════════════════════════════════════════════════════════════

def quality_filter(dialogue: Dict, min_avg_length: int = 80) -> bool:
    """
    多维度对话质量过滤。

    过滤条件：
      1. 对话轮次 >= 3（至少 3 轮 human+gpt）
      2. 平均回复长度 >= min_avg_length
      3. 无连续重复内容
      4. assistant 回复不能全是空话/套话
    """
    convs = dialogue.get("conversations", [])

    # 至少 3 轮完整对话
    human_turns = [c for c in convs if c["from"] == "human"]
    gpt_turns = [c for c in convs if c["from"] == "gpt"]

    if len(human_turns) < 3 or len(gpt_turns) < 3:
        return False

    # 平均回复长度
    avg_gpt_len = sum(len(c["value"]) for c in gpt_turns) / len(gpt_turns)
    if avg_gpt_len < min_avg_length:
        return False

    # 检查重复
    values = [c["value"][:100] for c in convs]
    if len(set(values)) < len(values) * 0.7:
        return False

    # 检查空话（过于简短或无实质内容的回复）
    short_replies = sum(1 for c in gpt_turns if len(c["value"]) < 30)
    if short_replies > len(gpt_turns) * 0.3:
        return False

    return True


# ══════════════════════════════════════════════════════════════════════
# 批量生成与导出
# ══════════════════════════════════════════════════════════════════════

def generate_multiturn_dataset(
    model: str = "qwen2.5:7b",
    num_dialogues: int = 200,
    max_turns: int = 6,
    min_turns: int = 3,
    output_dir: str = "./data/multiturn",
    output_name: str = "ecommerce_multiturn",
) -> str:
    """批量生成多轮对话数据集"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}.jsonl")
    checkpoint_path = os.path.join(output_dir, f"{output_name}_checkpoint.jsonl")

    # 加载已有数据（支持断点续传）
    existing = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
        logger.info(f"📌 从断点恢复：已有 {len(existing)} 条")

    generated = len(existing)
    all_dialogues = existing.copy()

    # 确保场景均匀分布
    scenario_cycle = MULTITURN_SCENARIOS * (num_dialogues // len(MULTITURN_SCENARIOS) + 1)
    random.shuffle(scenario_cycle)

    from tqdm import tqdm

    with tqdm(total=num_dialogues, initial=generated, desc="生成多轮对话") as pbar:
        for i in range(generated, num_dialogues):
            scenario = scenario_cycle[i % len(scenario_cycle)]

            dialogue = generate_multiturn_dialogue(
                scenario=scenario,
                model=model,
                max_turns=max_turns,
                min_turns=min_turns,
            )

            if dialogue and quality_filter(dialogue):
                all_dialogues.append(dialogue)
                pbar.update(1)

                # 每 10 条保存一次 checkpoint
                if len(all_dialogues) % 10 == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        for d in all_dialogues:
                            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 最终保存
    with open(output_path, "w", encoding="utf-8") as f:
        for d in all_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info(f"✅ 多轮对话数据集生成完成：{len(all_dialogues)} 条 → {output_path}")

    # 统计信息
    categories = {}
    total_turns = 0
    for d in all_dialogues:
        cat = d.get("metadata", {}).get("category", "未知")
        categories[cat] = categories.get(cat, 0) + 1
        total_turns += d.get("metadata", {}).get("num_turns", 0)

    logger.info(f"\n📊 数据集统计:")
    logger.info(f"  总对话数: {len(all_dialogues)}")
    logger.info(f"  平均轮次: {total_turns / max(len(all_dialogues), 1):.1f}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count} 条")

    return output_path


def convert_to_llamafactory_multiturn(
    input_path: str,
    output_dir: str = "./data/llamafactory_sft",
    system_prompt: str = "你是专业的电商运营和销售顾问，拥有丰富的实战经验。",
) -> str:
    """
    将生成的多轮对话数据转换为 LLaMA-Factory sharegpt 格式。
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ecommerce_multiturn.json")

    converted = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            convs = data.get("conversations", [])

            # 插入 system prompt
            formatted = [{"from": "system", "value": system_prompt}]
            for c in convs:
                formatted.append({"from": c["from"], "value": c["value"]})

            converted.append({"conversations": formatted})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ LLaMA-Factory 格式转换完成：{len(converted)} 条 → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="电商多轮对话数据生成")
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--num_dialogues", type=int, default=200)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--min_turns", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./data/multiturn")
    parser.add_argument("--llamafactory_dir", type=str, default="./data/llamafactory_sft")
    parser.add_argument("--skip_generate", action="store_true", help="跳过生成，只做格式转换")

    args = parser.parse_args()

    if not args.skip_generate:
        output_path = generate_multiturn_dataset(
            model=args.model,
            num_dialogues=args.num_dialogues,
            max_turns=args.max_turns,
            min_turns=args.min_turns,
            output_dir=args.output_dir,
        )
    else:
        output_path = os.path.join(args.output_dir, "ecommerce_multiturn.jsonl")

    # 转换为 LLaMA-Factory 格式
    if os.path.exists(output_path):
        convert_to_llamafactory_multiturn(
            input_path=output_path,
            output_dir=args.llamafactory_dir,
        )


if __name__ == "__main__":
    main()
