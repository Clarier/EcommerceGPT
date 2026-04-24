"""
电商大模型数据集生成器  v3.0 — 基于论文的 LLM 知识驱动版
============================================================
v3 核心思想：所有多样性来源于大模型自身的知识库 + 论文方法，
             不依赖人工预设的产品列表、模板列表或场景列表。

论文依据（按功能对应）：
┌────────────────────────────────────────────────────────────────────┐
│ [1] GLAN（Li et al., ICLR 2024）arXiv:2402.13064                   │
│     "Synthetic Data from Scratch: Generalized Instruction Tuning"  │
│     → 用 LLM 生成领域知识分类树（字段→子字段→学科→科目→教学大纲）  │
│     → 本项目：电商领域 → 功能模块 → 具体场景 → 细分任务            │
│                                                                      │
│ [2] Self-Instruct（Wang et al., ACL 2023）arXiv:2212.10560          │
│     "Aligning Language Models with Self-Generated Instructions"     │
│     → 从少量 seed 自举，用 LLM 持续生成新指令                       │
│     → 本项目：GLAN 分类树的叶节点作为种子，驱动指令生成             │
│                                                                      │
│ [3] InsTag（Lu et al., ICLR 2024）arXiv:2308.07074                  │
│     "#InsTag: Instruction Tagging for Analyzing SFT of LLMs"       │
│     → 用 LLM 为每条指令打开放集细粒度意图标签                       │
│     → 用标签覆盖率量化多样性，选取标签覆盖最广的子集                │
│                                                                      │
│ [4] WizardLM Evol-Instruct（Xu et al., ICLR 2024）arXiv:2304.12244 │
│     "Empowering LLMs to Follow Complex Instructions"                │
│     → In-Depth（增加约束/深化/具体化/推理步骤/复杂化输入）          │
│     → In-Breadth（突变生成同领域但更罕见的指令）                    │
│                                                                      │
│ [5] D3（Zhang et al., IJCAI 2025）arXiv:2503.11441                  │
│     "Diversity, Difficulty, Dependability-Aware Data Selection"     │
│     → 在淘宝直播场景验证（与本项目高度对齐）                        │
│     → Novelty 贪心选择 + ROUGE-L 去重（Self-Instruct 原版过滤法）   │
└────────────────────────────────────────────────────────────────────┘

输出：
  Stage 1 PT  → data/pretrain/ecommerce_pretrain.txt
  Stage 2 SFT → data/finetune/ecommerce_sft.jsonl
  Stage 3 DPO → data/reward/ecommerce_dpo.jsonl

用法：
  # 快速验证（~30分钟，本地 7B 模型）
  python generate_ecommerce_dataset_v3.py --model qwen2.5:7b --sft 200 --dpo 80

  # 标准训练（~3小时，14B 模型效果更好）
  python generate_ecommerce_dataset_v3.py --model qwen2.5:14b --sft 600 --dpo 250

  # 生成完毕后建议接 data_quality_pipeline.py 进一步清洗
"""

import json
import re
import random
import time
import argparse
import sys
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
# LLM 后端（Ollama）
# ══════════════════════════════════════════════════════════════════════

def _get_ollama():
    try:
        import ollama
        return ollama
    except ImportError:
        sys.exit("❌ 请先安装: pip install ollama tqdm")


class LLM:
    def __init__(self, model: str = "deepseek-chat"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key="sk-655ad4c737eb40239b5fa7813aef62a7",
            base_url="https://api.deepseek.com",
        )
        print(f"✅ 模型后端: {model} (DeepSeek API)")
        self._check_model()

    def _check_model(self):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "请回复OK"}],
                max_tokens=10,
                temperature=0.1,
            )
            content = resp.choices[0].message.content
            if content:
                print(f"  ✅ 模型连通性测试通过")
            else:
                print(f"  ⚠️ 模型返回空响应")
        except Exception as e:
            print(f"  ❌ 连接失败: {e}")

    def call(self, system: str, user: str,
             max_tokens: int = 1200, temperature: float = 0.85,
             json_mode: bool = False) -> str:
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ API 调用失败: {e}")
            return ""

    def call_retry(self, system: str, user: str,
                   max_tokens: int = 1200, temperature: float = 0.85,
                   json_mode: bool = False, retries: int = 3) -> str:
        for i in range(retries):
            result = self.call(system, user, max_tokens, temperature, json_mode)
            if result:
                return result
            import time
            time.sleep(2)
        return ""

    def _connectivity_test(self):
        """启动时做一次快速调用，确认模型可用"""
        print(f"  🔄 正在测试模型连通性...")
        try:
            result = self.call("你是助手。", "请回复OK", max_tokens=10, temperature=0.1)
            if result:
                print(f"  ✅ 模型连通性测试通过")
            else:
                print(f"  ⚠️ 模型返回空响应，请确认模型 [{self.model}] 已下载并可用")
                print(f"     尝试: ollama pull {self.model}")
        except Exception as e:
            print(f"  ❌ 模型连通性测试失败: {e}")
            print(f"     请确认: 1) ollama serve 已运行  2) ollama pull {self.model} 已完成")

    def _clean(self, text: str) -> str:
        for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "assistant\n"]:
            text = text.replace(tok, "")
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def parse_json(self, text: str) -> Optional[Union[dict, list]]:
        if not text:
            return None
        # 尝试从 markdown code block 中提取
        for pattern in [
            r"```(?:json)?\s*([\s\S]+?)```",
        ]:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except Exception:
                    pass
        # 直接解析
        try:
            return json.loads(text.strip())
        except Exception:
            pass
        # 尝试修复常见的 7B 模型 JSON 错误
        cleaned = text.strip()
        # 去除开头可能的非JSON文本（如 "以下是..."）
        for bracket in [("{", "}"), ("[", "]")]:
            s = cleaned.find(bracket[0])
            e = cleaned.rfind(bracket[1]) + 1
            if s >= 0 and e > s:
                fragment = cleaned[s:e]
                try:
                    return json.loads(fragment)
                except Exception:
                    pass
                # 修复尾部多余逗号: [..."item",] → [..."item"]
                fixed = re.sub(r',\s*([}\]])', r'\1', fragment)
                try:
                    return json.loads(fixed)
                except Exception:
                    pass
                # 修复单引号: {'key': 'val'} → {"key": "val"}
                fixed2 = fixed.replace("'", '"')
                try:
                    return json.loads(fixed2)
                except Exception:
                    pass
        return None


# ══════════════════════════════════════════════════════════════════════
# 技术1：GLAN 分类树（LLM 自主生成电商知识分类）
# ══════════════════════════════════════════════════════════════════════

class GLANTaxonomyBuilder:
    """
    论文：GLAN (Li et al., ICLR 2024) arXiv:2402.13064
    核心：用 LLM 半自动构建知识分类树
         Fields → Sub-fields → Disciplines → Subjects → Syllabus → Key Concepts

    针对电商领域的适配：
         功能模块 → 子模块 → 具体岗位/角色 → 核心任务 → 细分场景
    例：
         营销文案 → 商品文案 → 文案策划 → 标题撰写 → 爆款标题的关键词布局
         客户服务 → 售后处理 → 客服专员 → 差评应对 → 处理物流投诉的话术设计
    """

    # GLAN论文中LLM驱动分类树的prompt模板（电商域适配版）
    FIELD_PROMPT = """你是一位对中国电商行业有深刻理解的专家。
请将「中国电商营销与运营」领域的知识体系，分解为{n}个顶层功能模块（Fields）。

要求：
- 每个模块覆盖电商工作中真实存在的核心职能
- 模块间不重叠，合计覆盖电商从业者日常工作的完整场景
- 参考真实岗位分工（运营、客服、文案、数据、直播等）
- 只输出JSON数组：["模块1", "模块2", ...]
"""

    SUBFIELD_PROMPT = """你是一位对中国电商行业有深刻理解的专家。
「{field}」是电商营销与运营领域的一个核心模块。
请将它细分为{n}个子模块（Sub-fields），每个子模块代表该模块下的一个具体方向。

要求：
- 子模块要具体，是日常工作中真实存在的细分方向
- 避免过于抽象或宽泛
- 只输出JSON数组：["子模块1", "子模块2", ...]
"""

    TASK_PROMPT = """你是一位对中国电商行业有深刻理解的专家。
在「{field}」→「{subfield}」这个方向下，
请列出{n}个具体的、可以训练AI助手的独立任务或场景（Disciplines）。

要求：
- 每个任务要足够具体，可以直接生成一条SFT训练指令
- 覆盖不同难度（简单操作、中等分析、复杂决策）
- 结合真实的电商平台（淘宝/京东/拼多多/抖音/小红书）场景
- 只输出JSON数组：["任务1", "任务2", ...]
"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def build(self, n_fields: int = 8, n_subfields: int = 4,
              n_tasks: int = 4) -> list[dict]:
        """
        构建三层分类树，返回叶节点（具体任务）列表
        每个叶节点包含 field / subfield / task 三层路径
        理论任务总数 = n_fields × n_subfields × n_tasks
        """
        print(f"\n[GLAN] 构建电商知识分类树...")
        print(f"  目标: {n_fields}个模块 × {n_subfields}个子模块 × {n_tasks}个任务")
        print(f"  理论叶节点数: {n_fields * n_subfields * n_tasks}")

        # 第1层：功能模块（Fields）
        fields_text = self.llm.call_retry(
            system="你是电商行业专家，请严格按要求输出JSON。",
            user=self.FIELD_PROMPT.format(n=n_fields),
            max_tokens=400, temperature=0.7, json_mode=True,
        )
        fields = self.llm.parse_json(fields_text)
        if not isinstance(fields, list) or len(fields) < 2:
            # 降级处理
            fields = ["营销文案创作", "客户服务与运营", "数据分析与诊断",
                      "直播与内容运营", "促销活动策划", "商品运营管理",
                      "流量获取与广告", "用户运营与私域"]
        fields = fields[:n_fields]
        print(f"  ✅ 第1层 Fields: {fields}")

        # 第2+3层：子模块 + 具体任务
        all_leaves = []
        for field in tqdm(fields, desc="  构建分类树"):
            sub_text = self.llm.call_retry(
                system="你是电商行业专家，请严格按要求输出JSON。",
                user=self.SUBFIELD_PROMPT.format(field=field, n=n_subfields),
                max_tokens=300, temperature=0.8, json_mode=True,
            )
            subfields = self.llm.parse_json(sub_text)
            if not isinstance(subfields, list) or len(subfields) < 1:
                subfields = [f"{field}基础操作", f"{field}进阶策略"]
            subfields = subfields[:n_subfields]

            for subfield in subfields:
                task_text = self.llm.call_retry(
                    system="你是电商行业专家，请严格按要求输出JSON。",
                    user=self.TASK_PROMPT.format(
                        field=field, subfield=subfield, n=n_tasks),
                    max_tokens=400, temperature=0.85, json_mode=True,
                )
                tasks = self.llm.parse_json(task_text)
                if not isinstance(tasks, list) or len(tasks) < 1:
                    tasks = [f"{subfield}的核心操作方法"]
                tasks = tasks[:n_tasks]

                for task in tasks:
                    all_leaves.append({
                        "field": field,
                        "subfield": subfield,
                        "task": task,
                    })

        print(f"  ✅ 分类树构建完成，共 {len(all_leaves)} 个叶节点")
        return all_leaves


# ══════════════════════════════════════════════════════════════════════
# 技术2：Self-Instruct 风格的指令生成（从 GLAN 叶节点驱动）
# ══════════════════════════════════════════════════════════════════════

class SelfInstructGenerator:
    """
    论文：Self-Instruct (Wang et al., ACL 2023) arXiv:2212.10560
    核心："the more diverse the seed tasks are, the more diverse
          and better quality the generated tasks will be"

    流程：
    1. GLAN叶节点 → 作为多样化种子（替代人工写175个seed tasks）
    2. 以叶节点的 field/subfield/task 为上下文驱动LLM生成具体指令
    3. 生成时提供2个已有指令作为few-shot示例（Self-Instruct原版做法）
    """

    SFT_GEN_SYSTEM = """\
你是资深电商运营专家和销售顾问，精通淘宝、京东、拼多多、抖音等平台规则与营销技巧。

请生成一条高质量的电商场景指令微调训练数据，严格按以下JSON格式输出：
{
  "instruction": "具体的任务指令（40~120字，场景清晰，有明确的任务目标）",
  "input": "补充的上下文信息（若无则为空字符串\"\"）",
  "output": "专业、实用的回答（400~700字，有具体案例/话术示例/操作步骤，可操作性强）"
}

output质量要求：
- 给出具体话术示例、文案样本或操作步骤，不要泛泛而谈
- 结合电商平台实际规则和用户心理
- 有干货和细节，避免正确废话
"""

    def __init__(self, llm: LLM):
        self.llm = llm
        self._recent_instructions: list[str] = []  # 滚动窗口，few-shot用

    def generate_from_leaf(self, leaf: dict) -> Optional[dict]:
        """
        从 GLAN 分类树叶节点生成一条 SFT 样本
        leaf = {"field": ..., "subfield": ..., "task": ...}
        """
        # 构建 Self-Instruct 风格的 few-shot 上下文
        few_shot = ""
        if self._recent_instructions:
            examples = random.sample(
                self._recent_instructions,
                min(2, len(self._recent_instructions))
            )
            few_shot = "\n\n参考已有指令风格（不要重复）：\n" + \
                       "\n".join(f"- {e}" for e in examples)

        user = (
            f"请基于以下电商知识场景生成一条训练样本：\n"
            f"功能模块：{leaf['field']}\n"
            f"子模块：{leaf['subfield']}\n"
            f"具体任务场景：{leaf['task']}"
            f"{few_shot}"
        )

        text = self.llm.call_retry(
            system=self.SFT_GEN_SYSTEM, user=user,
            max_tokens=1200, temperature=0.85, json_mode=True,
        )
        data = self.llm.parse_json(text)

        if not isinstance(data, dict):
            return None
        if not all(k in data for k in ["instruction", "input", "output"]):
            return None
        if len(data.get("output", "")) < 150:
            return None
        if len(data.get("instruction", "")) < 15:
            return None

        # 更新滚动窗口（保持最近50条，用于 few-shot）
        instr = data["instruction"]
        self._recent_instructions.append(instr)
        if len(self._recent_instructions) > 50:
            self._recent_instructions.pop(0)

        # 附加分类信息（用于InsTag多样性分析）
        data["_taxonomy"] = {
            "field": leaf["field"],
            "subfield": leaf["subfield"],
            "task": leaf["task"],
        }
        return data


# ══════════════════════════════════════════════════════════════════════
# 技术3：InsTag 风格的意图标签 + 多样性度量
# ══════════════════════════════════════════════════════════════════════

class InsTagDiversitySelector:
    """
    论文：#InsTag (Lu et al., ICLR 2024) arXiv:2308.07074
    核心：
    1. 用 LLM 为每条指令生成细粒度意图标签（open-set，不预定义标签集）
    2. 多样性 = 数据集覆盖的标签种类数
    3. 选择策略：Complexity-first Diverse Sampling
       → 优先选复杂指令，同时保证标签覆盖的多样性

    本实现：
    - 标签生成：调用 LLM 为指令打标签
    - 多样性选择：基于标签覆盖的贪心选择（与 GLAN 分类树结合）
    - 对于离线场景：用 GLAN 分类树的 field/subfield 作为轻量级标签代理
    """

    TAG_SYSTEM = """\
你是一位NLP专家，请为以下电商指令生成3~5个细粒度意图标签。

标签要求（对应InsTag论文的open-set tagging）：
- 标签要捕捉指令的语义类型和意图（如：文案撰写、数据诊断、话术设计、策略规划等）
- 标签要足够具体，能区分不同类型的指令
- 标签格式：简洁的中文词组（2~6字）
- 只输出JSON数组：["标签1", "标签2", ...]
"""

    def __init__(self, llm: LLM, use_llm_tags: bool = True):
        self.llm = llm
        self.use_llm_tags = use_llm_tags

    def get_tags(self, sample: dict) -> list[str]:
        """
        为一条样本生成意图标签
        优先用LLM（完整InsTag），可回退到GLAN分类树标签（轻量代理）
        """
        # 方案A：LLM生成真实InsTag（质量更高）
        if self.use_llm_tags:
            instruction = sample.get("instruction", "")
            text = self.llm.call_retry(
                system=self.TAG_SYSTEM,
                user=f"指令：{instruction}",
                max_tokens=100, temperature=0.5, json_mode=True,
            )
            tags = self.llm.parse_json(text)
            if isinstance(tags, list) and len(tags) >= 1:
                return [t.strip() for t in tags if isinstance(t, str) and t.strip()]

        # 方案B：用GLAN分类树作为轻量代理标签
        taxonomy = sample.get("_taxonomy", {})
        tags = []
        if taxonomy.get("field"):
            tags.append(taxonomy["field"])
        if taxonomy.get("subfield"):
            tags.append(taxonomy["subfield"])
        return tags or ["未分类"]

    def diversity_select(self, samples: list[dict], budget: int,
                         tag_batch_size: int = 50) -> list[dict]:
        """
        InsTag论文的 Complexity-first Diverse Sampling：
        1. 按指令复杂度（长度 × 包含步骤数）降序排列
        2. 贪心选择：优先选择能引入新标签的样本
        3. 直到达到 budget 或样本用尽

        论文结论：6K InsTag选择的样本 > 52K随机数据（MT-Bench评估）
        """
        if not samples:
            return []

        # Step 1: 对前 tag_batch_size 条样本做 LLM 标注（其余用快速代理）
        print(f"  [InsTag] 为样本生成意图标签...")
        tagged = []
        for i, s in enumerate(tqdm(samples, desc="  打标签")):
            use_llm = i < tag_batch_size and self.use_llm_tags
            # 临时切换模式
            orig = self.use_llm_tags
            self.use_llm_tags = use_llm
            tags = self.get_tags(s)
            self.use_llm_tags = orig
            tagged.append((s, tags, self._complexity_score(s)))

        # Step 2: 按复杂度降序（Complexity-first）
        tagged.sort(key=lambda x: x[2], reverse=True)

        # Step 3: 贪心标签覆盖选择
        selected = []
        covered_tags: set[str] = set()

        for sample, tags, _ in tagged:
            if len(selected) >= budget:
                break
            new_tags = set(tags) - covered_tags
            # 能引入新标签，或者已选数量不足预算的50%时也选入
            if new_tags or len(selected) < budget * 0.5:
                selected.append(sample)
                covered_tags.update(tags)

        print(f"  [InsTag] 选择 {len(selected)} 条，覆盖标签类型: {len(covered_tags)}")
        return selected

    def _complexity_score(self, sample: dict) -> float:
        """
        InsTag 论文中 complexity = 指令包含的意图数量
        简化实现：指令长度 × 包含数字编号的句子数（推理步骤数代理）
        """
        instruction = sample.get("instruction", "")
        # 指令长度因子
        length_factor = min(len(instruction) / 100, 3.0)
        # 推理步骤因子（包含1.2.3.或①②③等编号）
        step_matches = len(re.findall(r'[①②③④⑤1-9][.、．]', instruction))
        # 包含条件约束的词
        constraint_words = ["如何", "策略", "分析", "方案", "同时", "并且",
                            "在...情况下", "当...时", "需要同时"]
        constraint_factor = sum(1 for w in constraint_words if w in instruction) * 0.3
        return length_factor + step_matches * 0.5 + constraint_factor


# ══════════════════════════════════════════════════════════════════════
# 技术4：Evol-Instruct（WizardLM，ICLR 2024）
# ══════════════════════════════════════════════════════════════════════

class EvolInstructAugmentor:
    """
    论文：WizardLM Evol-Instruct (Xu et al., ICLR 2024) arXiv:2304.12244
    核心：6种进化操作（5种 In-Depth + 1种 In-Breadth）
    目标：在已有 GLAN+Self-Instruct 生成的数据基础上，进一步增加多样性和复杂度

    论文参数：每条指令进化 M=4 轮（本地资源有限，默认 M=2）
    """

    OPERATIONS = {
        # In-Depth 1：增加约束
        "add_constraints": (
            "你是电商指令改写专家。将以下指令改写为更复杂版本：增加2~3个额外约束条件。\n"
            "要求：新指令合理可答，保持原场景，不说明做了什么改动，只输出新指令。\n\n"
            "原指令：{instruction}\n新指令："
        ),
        # In-Depth 2：深化
        "deepening": (
            "你是电商指令改写专家。将以下指令改写为更复杂版本：要求作答者提供更底层的原理分析或数据支撑。\n"
            "要求：新指令合理可答，保持原场景，只输出新指令。\n\n"
            "原指令：{instruction}\n新指令："
        ),
        # In-Depth 3：具体化
        "concretize": (
            "你是电商指令改写专家。将以下指令改写为更复杂版本：添加更具体的平台/品类/数据指标背景。\n"
            "要求：新指令合理可答，保持原场景，只输出新指令。\n\n"
            "原指令：{instruction}\n新指令："
        ),
        # In-Depth 4：增加推理步骤
        "increase_steps": (
            "你是电商指令改写专家。将以下指令改写为更复杂版本：要求按多个明确步骤（分析→诊断→方案→执行→评估）作答。\n"
            "要求：新指令合理可答，保持原场景，只输出新指令。\n\n"
            "原指令：{instruction}\n新指令："
        ),
        # In-Depth 5：复杂化输入
        "complicate_input": (
            "你是电商指令改写专家。将以下指令改写为更复杂版本：引入多个相互制约因素（预算/时间/竞争/团队等）。\n"
            "要求：新指令合理可答，保持原场景，只输出新指令。\n\n"
            "原指令：{instruction}\n新指令："
        ),
        # In-Breadth：突变（增加话题多样性）
        "breadth_mutation": (
            "你是电商指令创作专家。以下面指令为灵感，创造一个属于同一电商领域但更罕见、覆盖不同细分场景的全新指令。\n"
            "（来自WizardLM论文的In-Breadth Evolving：same domain but more rare）\n"
            "要求：新指令合理可答，长度难度相近，不提及原指令，只输出新指令。\n\n"
            "原指令：{instruction}\n新指令："
        ),
    }

    def __init__(self, llm: LLM):
        self.llm = llm
        self.ops = list(self.OPERATIONS.keys())

    def evolve(self, instruction: str) -> tuple[str, str]:
        """返回 (新指令, 操作名)"""
        op = random.choice(self.ops)
        prompt = self.OPERATIONS[op].format(instruction=instruction)
        new_instr = self.llm.call_retry(
            system="你是专业的电商指令改写专家。",
            user=prompt, max_tokens=250, temperature=0.85,
        )
        if new_instr and self._valid(instruction, new_instr):
            return new_instr.strip(), op
        return "", op

    def _valid(self, orig: str, evolved: str) -> bool:
        if len(evolved) < 10:
            return False
        if evolved.strip() == orig.strip():
            return False
        if any(w in evolved for w in ["原指令", "以下指令", "灵感来源"]):
            return False
        return True

    def augment_dataset(self, samples: list[dict],
                        rounds: int = 2) -> list[dict]:
        """
        对现有数据集做 Evol-Instruct 增强
        每轮：对所有样本随机进化，成功的生成新样本（需要后续补充 output）
        """
        new_samples = []
        current = [s["instruction"] for s in samples]

        for r in range(1, rounds + 1):
            print(f"  [Evol] 第 {r}/{rounds} 轮进化，种子池: {len(current)} 条")
            this_round = []
            for instr in tqdm(current, desc=f"  进化 R{r}"):
                new_instr, op = self.evolve(instr)
                if new_instr:
                    this_round.append({
                        "instruction": new_instr,
                        "input": "",
                        "output": "",  # 待填充
                        "_evol_op": op,
                        "_evol_round": r,
                    })
            new_samples.extend(this_round)
            current = [s["instruction"] for s in this_round]
            print(f"  [Evol] 本轮生成: {len(this_round)} 条")

        return new_samples


# ══════════════════════════════════════════════════════════════════════
# 技术5：D3 Novelty 去重过滤（Wang 2023 ROUGE-L + D3 novelty greedy）
# ══════════════════════════════════════════════════════════════════════

class NoveltyFilter:
    """
    结合两篇论文的过滤方法：
    - Self-Instruct（Wang et al., 2023）：ROUGE-L > 0.7 视为重复，过滤掉
    - D3（Zhang et al., IJCAI 2025）+ NovelSum：novelty 贪心选择
    """

    @staticmethod
    def _tokenize(text: str, max_len: int = 80) -> list:
        """中文友好分词：按字符切分（中文无空格，split()会失败）"""
        # 去除空白后按字符切分，对中文文本有效
        chars = list(text.replace(" ", "").replace("\n", ""))
        return chars[:max_len]

    def rouge_l(self, a: str, b: str) -> float:
        wa, wb = self._tokenize(a), self._tokenize(b)
        if not wa or not wb:
            return 0.0
        m, n = len(wa), len(wb)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if wa[i-1] == wb[j-1] else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        p, r = lcs / m, lcs / n
        return 2 * p * r / (p + r) if p + r else 0.0

    def ngram_novelty(self, cand: str, pool: list[str], n: int = 3) -> float:
        """NovelSum: novelty = 1 - max_similarity(cand, pool)"""
        def ngrams(t): return {t[i:i+n] for i in range(len(t) - n + 1)}
        cg = ngrams(cand.lower())
        if not cg or not pool:
            return 1.0
        sims = []
        for p in pool[-150:]:
            pg = ngrams(p.lower())
            if not pg:
                continue
            sims.append(len(cg & pg) / len(cg | pg))
        return 1.0 - max(sims) if sims else 1.0

    def filter_and_select(self, instructions: list[str],
                           budget: int,
                           rouge_threshold: float = 0.7,
                           novelty_threshold: float = 0.25) -> list[str]:
        """
        1. ROUGE-L 去重（Self-Instruct原版：阈值0.7）
        2. novelty贪心选择（D3/NovelSum）
        """
        # Step 1: ROUGE-L 去重
        deduped = []
        for cand in instructions:
            if not any(self.rouge_l(cand, e) > rouge_threshold for e in deduped[-80:]):
                deduped.append(cand)

        # Step 2: novelty 贪心
        random.shuffle(deduped)
        selected = []
        for cand in deduped:
            if len(selected) >= budget:
                break
            nov = self.ngram_novelty(cand, selected)
            if nov >= novelty_threshold or len(selected) < budget * 0.4:
                selected.append(cand)

        return selected


# ══════════════════════════════════════════════════════════════════════
# Stage 1：PT 预训练文章
# ══════════════════════════════════════════════════════════════════════

def build_pt_dataset(llm: LLM, taxonomy_leaves: list[dict],
                     num_articles: int, output_dir: str) -> list[str]:
    """
    PT文章主题直接从 GLAN 分类树的 field/subfield 层生成（不用人工预设主题）
    """
    print(f"\n{'━'*55}")
    print(f"  Stage 1 ▶ PT 预训练文本  |  数量={num_articles}")
    print(f"{'━'*55}")

    # 从分类树提取不重复的 field+subfield 组合作为文章主题
    seen = set()
    topics = []
    for leaf in taxonomy_leaves:
        key = f"{leaf['field']}—{leaf['subfield']}"
        if key not in seen:
            seen.add(key)
            topics.append((leaf['field'], leaf['subfield']))
    random.shuffle(topics)

    articles = []
    system = (
        "你是资深电商运营专家和营销顾问，有10年以上实战经验。"
        "请撰写一篇专业、实用、有深度的电商运营知识文章。"
        "要求：案例丰富、方法论清晰、可操作性强，字数900~1300字。"
        "直接输出正文，不要加文章标题行。"
    )

    for field, subfield in tqdm(topics[:num_articles], desc="  PT文章"):
        user = f"请撰写关于「{field}」中「{subfield}」方向的专业电商运营文章。"
        article = llm.call_retry(system=system, user=user,
                                 max_tokens=1600, temperature=0.75)
        if len(article) >= 300:
            articles.append(article)
        else:
            tqdm.write(f"    ⚠️ 文章过短，跳过: {field}/{subfield}")

    path = Path(output_dir) / "ecommerce_pretrain.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(articles), encoding="utf-8")
    print(f"  ✅ 保存 {len(articles)} 篇 → {path}")
    return articles


# ══════════════════════════════════════════════════════════════════════
# Stage 2：SFT 指令微调数据（GLAN + Self-Instruct + Evol + InsTag 联合）
# ══════════════════════════════════════════════════════════════════════

RESPONSE_SYSTEM = """\
你是资深电商运营专家和销售顾问，精通淘宝、京东、拼多多、抖音等平台规则与营销技巧。
请对以下电商场景问题给出专业、实用的回答。

回答要求：
1. 具体可执行的方案、话术示例或操作步骤
2. 结合平台规则和消费者心理
3. 有案例或数据参考
4. 逻辑清晰，可用数字编号
5. 字数 400~700字
"""


def build_sft_dataset(llm: LLM, taxonomy_leaves: list[dict],
                      num_samples: int, output_dir: str,
                      evol_rounds: int = 2,
                      use_instag: bool = True) -> list[dict]:
    """
    SFT 数据生成全流程：
    Phase A: GLAN + Self-Instruct → 初始样本（覆盖分类树所有叶节点）
    Phase B: Evol-Instruct → 增加复杂度和多样性
    Phase C: InsTag 多样性选择 → 保留最多样的子集
    Phase D: D3/Novelty 过滤 → 最终去重
    """
    print(f"\n{'━'*55}")
    print(f"  Stage 2 ▶ SFT 指令微调  |  目标={num_samples}")
    print(f"{'━'*55}")

    generator = SelfInstructGenerator(llm)
    evolver = EvolInstructAugmentor(llm)
    insTag = InsTagDiversitySelector(llm, use_llm_tags=use_instag)
    novelty_filter = NoveltyFilter()

    # ── Phase A：从 GLAN 叶节点生成初始指令 ──
    base_target = max(int(num_samples * 0.7), 50)
    checkpoint_path = Path(output_dir) / "_sft_checkpoint.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 尝试加载断点
    base_samples = []
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, encoding="utf-8") as cf:
                for line in cf:
                    base_samples.append(json.loads(line.strip()))
            print(f"  📌 从断点恢复: 已有 {len(base_samples)} 条，继续生成...")
        except Exception:
            base_samples = []

    print(f"\n  [Phase A] GLAN+Self-Instruct 生成初始指令 (目标约 {base_target} 条)")
    random.shuffle(taxonomy_leaves)
    # 循环遍历叶节点（如果叶节点数<目标，多轮遍历）
    failed = 0
    consecutive_failures = 0  # 连续失败计数，防止死循环
    with tqdm(total=base_target, initial=len(base_samples), desc="  Phase A") as pbar:
        idx = 0
        while len(base_samples) < base_target and idx < base_target * 3:
            if consecutive_failures >= 10:
                print(f"\n  ⚠️ 连续 {consecutive_failures} 次生成失败，"
                      f"可能是模型连接问题，请检查 Ollama 服务。")
                print(f"     已保存 {len(base_samples)} 条到断点文件，可重新运行继续。")
                break
            leaf = taxonomy_leaves[idx % len(taxonomy_leaves)]
            sample = generator.generate_from_leaf(leaf)
            if sample:
                base_samples.append(sample)
                pbar.update(1)
                consecutive_failures = 0
                # 每 20 条保存断点
                if len(base_samples) % 20 == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as cf:
                        for s in base_samples:
                            cf.write(json.dumps(s, ensure_ascii=False) + "\n")
            else:
                failed += 1
                consecutive_failures += 1
            idx += 1
    print(f"  Phase A 完成: {len(base_samples)} 条（跳过 {failed} 条）")
    # 清除断点文件
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ── Phase B：Evol-Instruct 增强 ──
    evol_target = num_samples - len(base_samples)
    if evol_target > 0 and evol_rounds > 0:
        print(f"\n  [Phase B] Evol-Instruct 增强（{evol_rounds}轮）")
        evol_shells = evolver.augment_dataset(base_samples, rounds=evol_rounds)
        # 为进化出的空白指令生成回复
        print(f"  为 {len(evol_shells)} 条进化指令生成回复...")
        evol_samples = []
        for shell in tqdm(evol_shells[:evol_target * 2], desc="  Phase B 生成回复"):
            if not shell["output"]:
                resp = llm.call_retry(
                    system=RESPONSE_SYSTEM,
                    user=shell["instruction"],
                    max_tokens=900, temperature=0.7,
                )
                if resp and len(resp) >= 100:
                    shell["output"] = resp
                    evol_samples.append(shell)
        print(f"  Phase B 完成: {len(evol_samples)} 条")
    else:
        evol_samples = []

    all_samples = base_samples + evol_samples
    print(f"\n  合并后总量: {len(all_samples)} 条")

    # ── Phase C：InsTag 多样性选择 ──
    print(f"\n  [Phase C] InsTag 多样性选择...")
    selected = insTag.diversity_select(all_samples, budget=num_samples)

    # ── Phase D：D3/Novelty 最终过滤 ──
    print(f"\n  [Phase D] Novelty 去重过滤...")
    instructions = [s["instruction"] for s in selected]
    clean_instrs = novelty_filter.filter_and_select(instructions, budget=num_samples)
    clean_set = set(clean_instrs)
    final_samples = [s for s in selected if s["instruction"] in clean_set]

    # 保存
    path = Path(output_dir) / "ecommerce_sft.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fields = ["instruction", "input", "output"]
    with open(path, "w", encoding="utf-8") as f:
        for s in final_samples:
            row = {k: s.get(k, "") for k in save_fields}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  ✅ SFT 最终保存: {len(final_samples)} 条 → {path}")
    return final_samples


# ══════════════════════════════════════════════════════════════════════
# Stage 3：DPO 偏好对比数据
# ══════════════════════════════════════════════════════════════════════

CHOSEN_SYSTEM = """\
你是顶级电商运营专家，拥有丰富实战经验。请提供一个高质量专业回答。

回答要求（全部满足）：
1. 具体可执行的方案或话术示例，不说废话
2. 结合电商平台规则和消费者心理
3. 有数据支撑或实际案例参考
4. 逻辑清晰，有层次感（可用1234或分段）
5. 字数 500~700字

直接输出回答，不要任何前缀。
"""

REJECTED_SYSTEM = """\
请为以下电商问题给出一个低质量回答，用于对比训练。

低质量回答特征（体现其中2~3个）：
- 过于笼统，没有具体话术或操作步骤
- 信息不完整，缺少关键要点
- 用大量正确废话填充，实际干货少
- 给出"多学习""多观察"等无用建议
- 忽视平台规则和实际约束
- 语言重复啰嗦，逻辑混乱

字数控制在 100~180字，看起来像是外行的敷衍回答。
直接输出回答内容。
"""

DPO_GEN_SYSTEM = """\
你是资深电商运营专家。请为以下电商场景生成一个高质量的DPO训练问题。

问题要求：
- 具体、有实际决策场景、难度适中
- 来自真实运营工作中会遇到的挑战
- 只输出JSON：{"question": "..."}
"""


def build_dpo_dataset(llm: LLM, sft_samples: list[dict],
                      taxonomy_leaves: list[dict],
                      num_samples: int, output_dir: str) -> list[dict]:
    """
    DPO 问题从两个来源生成（不用人工预设）：
    1. 从 SFT 样本的 instruction 直接复用（多样性已保证）
    2. 从 GLAN 分类树生成新的高质量问题（补充覆盖）
    """
    print(f"\n{'━'*55}")
    print(f"  Stage 3 ▶ DPO 偏好对比  |  数量={num_samples}")
    print(f"{'━'*55}")

    # 来源1：从SFT指令中提取（先打乱，取难度较高的）
    sft_questions = [s["instruction"] for s in sft_samples]
    # 优先选较长（较复杂）的指令作为DPO问题
    sft_questions.sort(key=len, reverse=True)
    questions = sft_questions[:int(num_samples * 0.6)]

    # 来源2：从GLAN分类树生成新问题（补充不同场景）
    need_extra = num_samples - len(questions)
    if need_extra > 0:
        print(f"  从GLAN分类树生成额外 {need_extra} 个问题...")
        random.shuffle(taxonomy_leaves)
        extra_qs = []
        for leaf in tqdm(taxonomy_leaves, desc="  生成DPO问题"):
            if len(extra_qs) >= need_extra * 2:
                break
            user = (
                f"电商场景：{leaf['field']} → {leaf['subfield']} → {leaf['task']}\n"
                f"请生成一个适合DPO对比训练的高质量问题。"
            )
            text = llm.call_retry(
                system=DPO_GEN_SYSTEM, user=user,
                max_tokens=150, temperature=0.9, json_mode=True,
            )
            data = llm.parse_json(text)
            if isinstance(data, dict) and "question" in data:
                q = data["question"].strip()
                if len(q) >= 15:
                    extra_qs.append(q)
        questions.extend(extra_qs[:need_extra])

    random.shuffle(questions)
    questions = questions[:num_samples]
    print(f"  DPO问题总数: {len(questions)}")

    # 生成 chosen/rejected 对
    records = []
    failed = 0
    for q in tqdm(questions, desc="  生成DPO对"):
        chosen = llm.call_retry(
            system=CHOSEN_SYSTEM, user=f"问题：{q}",
            max_tokens=900, temperature=0.65,
        )
        rejected = llm.call_retry(
            system=REJECTED_SYSTEM, user=f"问题：{q}",
            max_tokens=280, temperature=0.9,
        )
        if not chosen or not rejected:
            failed += 1
            continue
        if len(chosen) < 200 or len(rejected) < 80:
            failed += 1
            continue
        if len(chosen) < len(rejected) * 1.8:
            failed += 1
            continue
        records.append({
            "instruction": q, "input": "",
            "output": [chosen, rejected],
        })

    path = Path(output_dir) / "ecommerce_dpo.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✅ 保存 {len(records)} 对（跳过 {failed}）→ {path}")
    return records


# ══════════════════════════════════════════════════════════════════════
# 质量报告
# ══════════════════════════════════════════════════════════════════════

def print_report(taxonomy_leaves, pt_data, sft_data, dpo_data):
    print(f"\n{'━'*55}")
    print(f"  📊 数据集质量报告  （v3.0 LLM知识驱动）")
    print(f"{'━'*55}")

    fields = list({l["field"] for l in taxonomy_leaves})
    subfields = list({l["subfield"] for l in taxonomy_leaves})
    print(f"\n  🗂  GLAN 分类树")
    print(f"     功能模块数:   {len(fields)}")
    print(f"     子模块数:     {len(subfields)}")
    print(f"     叶节点总数:   {len(taxonomy_leaves)}")
    print(f"     模块列表: {', '.join(fields)}")

    if pt_data:
        lens = [len(a) for a in pt_data]
        print(f"\n  📄 Stage1 PT")
        print(f"     文章数:    {len(pt_data)}")
        print(f"     平均字数:  {sum(lens)//len(lens)}")

    if sft_data:
        out_l = [len(s.get("output","")) for s in sft_data]
        ins_l = [len(s.get("instruction","")) for s in sft_data]
        # 统计进化样本占比
        evol_cnt = sum(1 for s in sft_data if s.get("_evol_op"))
        print(f"\n  📋 Stage2 SFT")
        print(f"     样本数:        {len(sft_data)}")
        print(f"     其中进化样本:  {evol_cnt} ({evol_cnt/max(len(sft_data),1)*100:.0f}%)")
        print(f"     平均指令长度:  {sum(ins_l)//len(ins_l)} 字")
        print(f"     平均回答长度:  {sum(out_l)//len(out_l)} 字")
        # InsTag 覆盖的分类模块
        covered = {s.get("_taxonomy", {}).get("field") for s in sft_data if s.get("_taxonomy")}
        print(f"     覆盖功能模块:  {len(covered)} 个")

    if dpo_data:
        valid = [d for d in dpo_data
                 if isinstance(d.get("output"), list) and len(d["output"]) == 2]
        if valid:
            c = [len(d["output"][0]) for d in valid]
            r = [len(d["output"][1]) for d in valid]
            ratio = (sum(c)/len(c)) / (sum(r)/len(r))
            tag = "✅ 优秀" if ratio >= 3 else ("✅ 良好" if ratio >= 2 else "⚠️ 偏低")
            print(f"\n  ⚖️  Stage3 DPO")
            print(f"     样本对数:     {len(valid)}")
            print(f"     chosen均长:   {sum(c)//len(c)} 字")
            print(f"     rejected均长: {sum(r)//len(r)} 字")
            print(f"     质量比:       {ratio:.1f}x  {tag}")

    print(f"\n{'━'*55}")
    print(f"  ✅ 多样性来源：GLAN分类树 + Self-Instruct自举")
    print(f"               + Evol-Instruct进化 + InsTag过滤")
    print(f"  📑 论文依据：GLAN(ICLR2024) | WizardLM(ICLR2024)")
    print(f"               Self-Instruct(ACL2023) | InsTag(ICLR2024)")
    print(f"               D3(IJCAI2025) | NovelSum(2024)")
    print(f"{'━'*55}\n")


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="电商数据集生成器 v3.0 — LLM知识驱动（基于GLAN/Evol-Instruct/InsTag/D3）"
    )
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--stages", type=str, default="1,2,3")

    # GLAN 分类树参数
    parser.add_argument("--glan_fields",    type=int, default=8,
                        help="GLAN顶层功能模块数（Fields，论文建议>=5）")
    parser.add_argument("--glan_subfields", type=int, default=4,
                        help="每个模块的子模块数（Sub-fields）")
    parser.add_argument("--glan_tasks",     type=int, default=4,
                        help="每个子模块的具体任务数（Disciplines）")

    # 数据量参数
    parser.add_argument("--pt",  type=int, default=20, dest="pt_articles")
    parser.add_argument("--sft", type=int, default=500, dest="sft_samples")
    parser.add_argument("--dpo", type=int, default=200, dest="dpo_samples")

    # 算法参数
    parser.add_argument("--evol_rounds", type=int, default=2,
                        help="Evol-Instruct进化轮次（WizardLM论文M=4，本地建议2）")
    parser.add_argument("--no_instag",   action="store_true",
                        help="禁用InsTag LLM打标签（改用GLAN代理标签，速度更快）")

    # 输出目录
    parser.add_argument("--pt_dir",  type=str, default="data/pretrain")
    parser.add_argument("--sft_dir", type=str, default="data/finetune")
    parser.add_argument("--dpo_dir", type=str, default="data/reward")

    # 是否保存分类树
    parser.add_argument("--save_taxonomy", type=str, default="data/taxonomy.json",
                        help="保存GLAN分类树（方便复用，下次跳过构建）")
    parser.add_argument("--load_taxonomy", type=str, default="",
                        help="加载已有GLAN分类树（跳过构建步骤）")

    args = parser.parse_args()
    stages = [int(s.strip()) for s in args.stages.split(",")]

    print(f"\n{'━'*60}")
    print(f"  🛒 电商数据集生成器 v3.0 — LLM知识驱动")
    print(f"  模型: {args.model}")
    print(f"{'━'*60}")
    print(f"  论文技术栈:")
    print(f"    GLAN       分类树自动构建（ICLR 2024）")
    print(f"    Self-Instruct 指令自举生成（ACL 2023）")
    print(f"    Evol-Instruct 复杂度进化（ICLR 2024）")
    print(f"    InsTag     意图标签多样性过滤（ICLR 2024）")
    print(f"    D3/NovelSum  novelty去重（IJCAI 2025 / 2024）")

    llm = LLM(model=args.model)

    # ── 构建或加载 GLAN 分类树 ──
    if args.load_taxonomy and Path(args.load_taxonomy).exists():
        print(f"\n[GLAN] 加载已有分类树: {args.load_taxonomy}")
        with open(args.load_taxonomy, encoding="utf-8") as f:
            taxonomy_leaves = json.load(f)
        print(f"  加载 {len(taxonomy_leaves)} 个叶节点")
    else:
        glan = GLANTaxonomyBuilder(llm)
        taxonomy_leaves = glan.build(
            n_fields=args.glan_fields,
            n_subfields=args.glan_subfields,
            n_tasks=args.glan_tasks,
        )
        if args.save_taxonomy:
            Path(args.save_taxonomy).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_taxonomy, "w", encoding="utf-8") as f:
                json.dump(taxonomy_leaves, f, ensure_ascii=False, indent=2)
            print(f"  分类树已保存 → {args.save_taxonomy}")

    pt_data, sft_data, dpo_data = [], [], []

    if 1 in stages:
        pt_data = build_pt_dataset(llm, taxonomy_leaves,args.pt_articles, args.pt_dir)

    if 2 in stages:
        sft_data = build_sft_dataset(
            llm, taxonomy_leaves, args.sft_samples, args.sft_dir,
            evol_rounds=args.evol_rounds,
            use_instag=not args.no_instag,
        )

    if 3 in stages:
        dpo_data = build_dpo_dataset(
            llm, sft_data or [], taxonomy_leaves,
            args.dpo_samples, args.dpo_dir,
        )

    print_report(taxonomy_leaves, pt_data, sft_data, dpo_data)


if __name__ == "__main__":
    main()
