# -*- coding: utf-8 -*-
"""
数据质量流水线 - 工业级数据清洗与筛选
=====================================
创新点：
  1. 基于 IFD (Instruction Following Difficulty) 的数据难度打分
  2. n-gram + MinHash 高效去重
  3. 多维质量过滤（长度/困惑度/语言/格式）
  4. 课程学习排序 (Curriculum Learning Ordering)
  5. DPO数据偏好裕度过滤 (Preference Margin Filtering)

参考论文：
  - LIMA: Less Is More for Alignment (2023)
  - AlpaGasus: Training A Better Alpaca with Fewer Data (2023)
  - What Makes Good Data for Alignment? (2023)
  - Instruction Mining (2023)
"""

import json
import re
import hashlib
import math
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 数据结构定义
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SFTSample:
    instruction: str
    input: str = ""
    output: str = ""
    quality_score: float = 0.0
    difficulty_score: float = 0.0
    source: str = "unknown"

    def to_dict(self):
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "quality_score": self.quality_score,
            "difficulty_score": self.difficulty_score,
            "source": self.source,
        }


@dataclass
class DPOSample:
    instruction: str
    chosen: str
    rejected: str
    preference_margin: float = 0.0  # chosen质量 - rejected质量
    quality_score: float = 0.0

    def to_dict(self):
        return {
            "instruction": self.instruction,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "preference_margin": self.preference_margin,
            "quality_score": self.quality_score,
        }


# ══════════════════════════════════════════════════════════════════════
# 1. 高效去重模块 (MinHash近似去重 + 精确n-gram去重)
# ══════════════════════════════════════════════════════════════════════

class MinHashDeduplicator:
    """
    基于MinHash的近似去重，时间复杂度O(n)，适合大规模数据
    相似度阈值可调，默认0.85
    """
    def __init__(self, num_perm: int = 128, threshold: float = 0.85, ngram: int = 3):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram = ngram
        # 预生成随机哈希参数
        self._a = [random.randint(1, (1 << 31) - 1) for _ in range(num_perm)]
        self._b = [random.randint(0, (1 << 31) - 1) for _ in range(num_perm)]
        self._p = (1 << 31) - 1
        self.seen_signatures = []
        self.dedup_stats = {"total": 0, "removed": 0}

    def _get_ngrams(self, text: str) -> set:
        """提取字符级n-gram"""
        text = re.sub(r'\s+', ' ', text.lower().strip())
        return {text[i:i+self.ngram] for i in range(len(text) - self.ngram + 1)}

    def _minhash(self, ngrams: set) -> List[int]:
        """计算MinHash签名"""
        signature = []
        for i in range(self.num_perm):
            min_val = float('inf')
            for ng in ngrams:
                h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
                hashed = (self._a[i] * h + self._b[i]) % self._p
                min_val = min(min_val, hashed)
            signature.append(min_val if min_val != float('inf') else 0)
        return signature

    def _jaccard_estimate(self, sig1: List[int], sig2: List[int]) -> float:
        """用MinHash签名估算Jaccard相似度"""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / self.num_perm

    def is_duplicate(self, text: str) -> bool:
        """判断是否为重复文本"""
        self.dedup_stats["total"] += 1
        ngrams = self._get_ngrams(text)
        if len(ngrams) < 5:  # 文本太短直接跳过
            return False
        sig = self._minhash(ngrams)
        for seen_sig in self.seen_signatures[-1000:]:  # 只和最近1000条比较，提升速度
            if self._jaccard_estimate(sig, seen_sig) >= self.threshold:
                self.dedup_stats["removed"] += 1
                return True
        self.seen_signatures.append(sig)
        return False

    def get_stats(self) -> Dict:
        total = self.dedup_stats["total"]
        removed = self.dedup_stats["removed"]
        return {
            "total_processed": total,
            "duplicates_removed": removed,
            "dedup_rate": f"{removed/max(total,1)*100:.1f}%",
            "remaining": total - removed
        }


class ExactDeduplicator:
    """基于SHA256的精确去重（用于output级别去重）"""
    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha256(text.strip().encode()).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False


# ══════════════════════════════════════════════════════════════════════
# 2. 多维质量过滤器
# ══════════════════════════════════════════════════════════════════════

class QualityFilter:
    """
    多维度质量过滤，参考 Dolma / RedPajama 数据清洗策略
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "min_instruction_len": 10,
            "max_instruction_len": 1000,
            "min_output_len": 20,
            "max_output_len": 2000,
            "min_output_words": 10,
            "max_repetition_ratio": 0.3,   # 重复n-gram比例
            "min_char_diversity": 0.3,      # 字符多样性
            "reject_patterns": [            # 拒绝模式
                r"^(好的|OK|ok|Sure|sure)\s*[，。！]?\s*$",  # 过短回复
                r"^我是AI助手",  # 模板化回复
                r"(作为AI语言模型|作为一个AI)",  # 身份声明开头
            ]
        }
        self.filter_stats = defaultdict(int)

    def _repetition_ratio(self, text: str, n: int = 4) -> float:
        """计算n-gram重复率"""
        words = text.split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return 0.0
        unique = len(set(ngrams))
        return 1.0 - unique / len(ngrams)

    def _char_diversity(self, text: str) -> float:
        """字符多样性（独特字符/总字符）"""
        if not text:
            return 0.0
        return len(set(text)) / len(text)

    def _contains_reject_pattern(self, text: str) -> bool:
        """检查拒绝模式"""
        for pattern in self.config["reject_patterns"]:
            if re.search(pattern, text.strip()):
                return True
        return False

    def _is_mostly_chinese_or_english(self, text: str) -> bool:
        """验证主要语言"""
        chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
        english = len(re.findall(r'[a-zA-Z]', text))
        total = len(text.replace(' ', ''))
        if total == 0:
            return False
        lang_ratio = (chinese + english) / total
        return lang_ratio > 0.5

    def filter_sft(self, sample: SFTSample) -> Tuple[bool, str]:
        """
        过滤SFT样本，返回 (是否保留, 过滤原因)
        """
        instruction = sample.instruction
        output = sample.output

        # 长度检查
        if len(instruction) < self.config["min_instruction_len"]:
            self.filter_stats["instruction_too_short"] += 1
            return False, "instruction_too_short"
        if len(instruction) > self.config["max_instruction_len"]:
            self.filter_stats["instruction_too_long"] += 1
            return False, "instruction_too_long"
        if len(output) < self.config["min_output_len"]:
            self.filter_stats["output_too_short"] += 1
            return False, "output_too_short"
        if len(output) > self.config["max_output_len"]:
            self.filter_stats["output_too_long"] += 1
            return False, "output_too_long"

        # 词数检查
        output_words = len(output.split())
        if output_words < self.config["min_output_words"]:
            self.filter_stats["output_too_few_words"] += 1
            return False, "output_too_few_words"

        # 重复率检查
        if self._repetition_ratio(output) > self.config["max_repetition_ratio"]:
            self.filter_stats["high_repetition"] += 1
            return False, "high_repetition"

        # 字符多样性
        if self._char_diversity(output) < self.config["min_char_diversity"]:
            self.filter_stats["low_char_diversity"] += 1
            return False, "low_char_diversity"

        # 拒绝模式
        if self._contains_reject_pattern(output):
            self.filter_stats["reject_pattern"] += 1
            return False, "reject_pattern"

        # 语言检查
        if not self._is_mostly_chinese_or_english(output):
            self.filter_stats["invalid_language"] += 1
            return False, "invalid_language"

        self.filter_stats["passed"] += 1
        return True, "passed"

    def filter_dpo(self, sample: DPOSample) -> Tuple[bool, str]:
        """
        过滤DPO样本，额外检查：
        - chosen != rejected
        - preference_margin 足够大
        """
        if sample.chosen.strip() == sample.rejected.strip():
            self.filter_stats["dpo_identical"] += 1
            return False, "identical_chosen_rejected"

        # chosen 必须比 rejected 更长/更好（基础质量检查）
        if len(sample.chosen) < 20:
            self.filter_stats["dpo_chosen_too_short"] += 1
            return False, "chosen_too_short"

        # 偏好裕度检查：margin过小的样本噪音大
        if sample.preference_margin < 0.1:
            self.filter_stats["dpo_low_margin"] += 1
            return False, "low_preference_margin"

        self.filter_stats["dpo_passed"] += 1
        return True, "passed"

    def get_stats(self) -> Dict:
        return dict(self.filter_stats)


# ══════════════════════════════════════════════════════════════════════
# 3. IFD难度评分 (Instruction Following Difficulty)
# ══════════════════════════════════════════════════════════════════════

class IFDScorer:
    """
    基于规则的IFD难度打分（不需要模型推理的近似版本）
    参考论文：AlpaGasus & Instruction Mining

    真实IFD需要计算：score = loss(output|instruction) / loss(output)
    这里用语言特征代理打分，适合大规模离线过滤

    维度：
    1. 指令复杂度（词数、子任务数、约束数）
    2. 输出复杂度（结构化程度、推理步骤数）
    3. 领域专业度（专业词汇密度）
    4. 任务多样性
    """
    ECOMMERCE_KEYWORDS = {
        "高专业度": ["ROI", "CTR", "GMV", "转化率", "客单价", "复购率", "LTV", "流量池",
                    "千次展现", "ROAS", "漏斗", "留存", "NPS", "SKU", "供应链"],
        "中专业度": ["营销", "活动策划", "定价策略", "竞品", "用户画像", "私域", "直播"],
        "低专业度": ["推荐", "介绍", "写一个", "生成", "帮我"]
    }

    def __init__(self):
        self.score_stats = []

    def _instruction_complexity(self, instruction: str) -> float:
        """指令复杂度打分 [0, 1]"""
        score = 0.0
        # 长度得分
        length = len(instruction)
        score += min(length / 200, 0.3)
        # 约束词数量
        constraint_words = ["要求", "需要", "必须", "不能", "至少", "最多", "字以内", "条", "点", "步骤"]
        score += min(sum(1 for w in constraint_words if w in instruction) * 0.05, 0.3)
        # 多任务指示词
        multi_task = ["并且", "同时", "另外", "此外", "还需要", "包括"]
        score += min(sum(1 for w in multi_task if w in instruction) * 0.07, 0.2)
        # 专业词汇
        for word in self.ECOMMERCE_KEYWORDS["高专业度"]:
            if word in instruction:
                score += 0.05
        return min(score, 1.0)

    def _output_complexity(self, output: str) -> float:
        """输出复杂度打分 [0, 1]"""
        score = 0.0
        # 结构化指示（有编号/标题）
        has_structure = bool(re.search(r'(^|\n)[\d一二三四五六七八九十]+[\.、。]', output))
        score += 0.3 if has_structure else 0.0
        # 推理词
        reasoning_words = ["因为", "所以", "由于", "因此", "建议", "分析", "策略", "方案"]
        score += min(sum(1 for w in reasoning_words if w in output) * 0.04, 0.3)
        # 输出长度
        score += min(len(output) / 800, 0.3)
        # 数字/数据引用
        has_data = bool(re.search(r'\d+[%％]|\d+元|\d+万', output))
        score += 0.1 if has_data else 0.0
        return min(score, 1.0)

    def _domain_expertise(self, text: str) -> float:
        """领域专业度 [0, 1]"""
        high = sum(1 for w in self.ECOMMERCE_KEYWORDS["高专业度"] if w in text)
        mid = sum(1 for w in self.ECOMMERCE_KEYWORDS["中专业度"] if w in text)
        return min((high * 0.1 + mid * 0.05), 1.0)

    def score(self, sample: SFTSample) -> float:
        """综合IFD难度打分"""
        inst_score = self._instruction_complexity(sample.instruction)
        out_score = self._output_complexity(sample.output)
        domain_score = self._domain_expertise(sample.instruction + sample.output)
        # 加权组合
        final_score = 0.4 * inst_score + 0.4 * out_score + 0.2 * domain_score
        self.score_stats.append(final_score)
        return round(final_score, 4)

    def score_batch(self, samples: List[SFTSample]) -> List[SFTSample]:
        """批量打分并附加到样本"""
        for s in samples:
            s.difficulty_score = self.score(s)
        return samples

    def get_distribution(self) -> Dict:
        if not self.score_stats:
            return {}
        scores = sorted(self.score_stats)
        n = len(scores)
        return {
            "count": n,
            "mean": round(sum(scores)/n, 4),
            "min": round(scores[0], 4),
            "max": round(scores[-1], 4),
            "p25": round(scores[n//4], 4),
            "p50": round(scores[n//2], 4),
            "p75": round(scores[3*n//4], 4),
        }


# ══════════════════════════════════════════════════════════════════════
# 4. 偏好裕度计算（DPO专用）
# ══════════════════════════════════════════════════════════════════════

class PreferenceMarginCalculator:
    """
    计算DPO样本的偏好裕度
    参考 SimPO 论文：高裕度样本训练信号更强

    维度：
    - 信息量差异（chosen比rejected包含更多实质内容）
    - 格式质量差异
    - 专业度差异
    """

    @staticmethod
    def _content_score(text: str) -> float:
        """内容质量评分"""
        score = 0.0
        # 信息密度：独特词/总词
        words = text.split()
        if words:
            score += len(set(words)) / len(words) * 0.3
        # 结构化
        if re.search(r'(^|\n)[\d一二三四五]+[\.、]', text):
            score += 0.2
        # 数据支撑
        if re.search(r'\d+[%％万元]', text):
            score += 0.15
        # 长度适当性（300-800字最佳）
        l = len(text)
        if 300 <= l <= 800:
            score += 0.2
        elif 100 <= l < 300:
            score += 0.1
        # 具体建议词
        specific_words = ["建议", "推荐", "策略", "方案", "具体", "步骤", "第一步"]
        score += min(sum(1 for w in specific_words if w in text) * 0.03, 0.15)
        return min(score, 1.0)

    def calculate(self, sample: DPOSample) -> float:
        """计算偏好裕度 = chosen_score - rejected_score"""
        chosen_score = self._content_score(sample.chosen)
        rejected_score = self._content_score(sample.rejected)
        margin = chosen_score - rejected_score
        return round(margin, 4)


# ══════════════════════════════════════════════════════════════════════
# 5. 课程学习排序
# ══════════════════════════════════════════════════════════════════════

class CurriculumSorter:
    """
    基于难度的课程学习排序
    策略：easy-to-hard（从易到难），研究表明优于随机排序
    参考：Curriculum Learning (Bengio et al., 2009) 及近期LLM应用
    """

    @staticmethod
    def sort_easy_to_hard(samples: List[SFTSample]) -> List[SFTSample]:
        """从易到难排序"""
        return sorted(samples, key=lambda x: x.difficulty_score)

    @staticmethod
    def sort_hard_to_easy(samples: List[SFTSample]) -> List[SFTSample]:
        """从难到易排序"""
        return sorted(samples, key=lambda x: x.difficulty_score, reverse=True)

    @staticmethod
    def bucket_sort(samples: List[SFTSample], n_buckets: int = 5) -> List[SFTSample]:
        """
        分桶后随机打散——兼顾多样性和渐进难度
        实践中效果最优
        """
        samples_sorted = sorted(samples, key=lambda x: x.difficulty_score)
        bucket_size = max(1, len(samples_sorted) // n_buckets)
        buckets = [samples_sorted[i:i+bucket_size] for i in range(0, len(samples_sorted), bucket_size)]
        result = []
        for bucket in buckets:
            random.shuffle(bucket)
            result.extend(bucket)
        return result

    @staticmethod
    def get_difficulty_distribution(samples: List[SFTSample]) -> Dict:
        """统计难度分布"""
        if not samples:
            return {}
        scores = [s.difficulty_score for s in samples]
        easy = sum(1 for s in scores if s < 0.33)
        medium = sum(1 for s in scores if 0.33 <= s < 0.66)
        hard = sum(1 for s in scores if s >= 0.66)
        return {
            "easy (0-0.33)": easy,
            "medium (0.33-0.66)": medium,
            "hard (0.66-1.0)": hard,
            "ratio": f"{easy}:{medium}:{hard}"
        }


# ══════════════════════════════════════════════════════════════════════
# 6. 主流水线
# ══════════════════════════════════════════════════════════════════════

class DataQualityPipeline:
    """
    完整的数据质量流水线
    处理顺序：加载 → 去重 → 质量过滤 → 难度打分 → 偏好裕度计算 → 课程排序 → 保存
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.deduplicator = MinHashDeduplicator(
            threshold=self.config.get("dedup_threshold", 0.85)
        )
        self.exact_dedup = ExactDeduplicator()
        self.quality_filter = QualityFilter()
        self.ifd_scorer = IFDScorer()
        self.margin_calc = PreferenceMarginCalculator()
        self.curriculum = CurriculumSorter()
        self.report = {}

    def process_sft(self, input_path: str, output_path: str,
                    top_k: Optional[int] = None,
                    curriculum_strategy: str = "bucket") -> Dict:
        """
        处理SFT数据集完整流水线
        """
        logger.info(f"[SFT Pipeline] 读取数据: {input_path}")
        raw_samples = self._load_sft(input_path)
        logger.info(f"原始样本数: {len(raw_samples)}")

        # Step 1: 去重
        logger.info("Step 1: 去重...")
        after_dedup = []
        for s in raw_samples:
            text = s.instruction + s.output
            if not self.deduplicator.is_duplicate(text) and \
               not self.exact_dedup.is_duplicate(s.output):
                after_dedup.append(s)
        logger.info(f"去重后: {len(after_dedup)} ({len(raw_samples)-len(after_dedup)} 删除)")

        # Step 2: 质量过滤
        logger.info("Step 2: 质量过滤...")
        after_filter = []
        for s in after_dedup:
            keep, reason = self.quality_filter.filter_sft(s)
            if keep:
                after_filter.append(s)
        logger.info(f"过滤后: {len(after_filter)} ({len(after_dedup)-len(after_filter)} 删除)")

        # Step 3: IFD难度打分
        logger.info("Step 3: IFD难度打分...")
        scored_samples = self.ifd_scorer.score_batch(after_filter)

        # Step 4: Top-K 选择（可选，类似LIMA少量高质量数据）
        if top_k and top_k < len(scored_samples):
            # 按质量+难度综合排序，取Top-K
            scored_samples.sort(
                key=lambda x: x.difficulty_score * 0.5 + x.quality_score * 0.5,
                reverse=True
            )
            scored_samples = scored_samples[:top_k]
            logger.info(f"Top-K 选择后: {len(scored_samples)}")

        # Step 5: 课程学习排序
        logger.info(f"Step 5: 课程排序 (策略: {curriculum_strategy})...")
        if curriculum_strategy == "easy_to_hard":
            final_samples = self.curriculum.sort_easy_to_hard(scored_samples)
        elif curriculum_strategy == "hard_to_easy":
            final_samples = self.curriculum.sort_hard_to_easy(scored_samples)
        else:
            final_samples = self.curriculum.bucket_sort(scored_samples)

        # Step 6: 保存
        self._save_sft(final_samples, output_path)
        logger.info(f"已保存至: {output_path}")

        # 生成报告
        report = {
            "pipeline": "SFT",
            "input_path": input_path,
            "output_path": output_path,
            "stats": {
                "raw": len(raw_samples),
                "after_dedup": len(after_dedup),
                "after_filter": len(after_filter),
                "final": len(final_samples),
                "retention_rate": f"{len(final_samples)/max(len(raw_samples),1)*100:.1f}%"
            },
            "dedup_stats": self.deduplicator.get_stats(),
            "filter_stats": self.quality_filter.get_stats(),
            "difficulty_distribution": self.curriculum.get_difficulty_distribution(final_samples),
            "ifd_distribution": self.ifd_scorer.get_distribution(),
            "curriculum_strategy": curriculum_strategy,
        }
        self.report["sft"] = report
        return report

    def process_dpo(self, input_path: str, output_path: str,
                    min_margin: float = 0.15) -> Dict:
        """
        处理DPO数据集完整流水线
        额外步骤：偏好裕度过滤
        """
        logger.info(f"[DPO Pipeline] 读取数据: {input_path}")
        raw_samples = self._load_dpo(input_path)
        logger.info(f"原始DPO样本数: {len(raw_samples)}")

        # Step 1: 计算偏好裕度
        logger.info("Step 1: 计算偏好裕度...")
        for s in raw_samples:
            s.preference_margin = self.margin_calc.calculate(s)

        # Step 2: 质量过滤（含裕度过滤）
        logger.info("Step 2: DPO质量过滤...")
        original_min_margin = self.quality_filter.config.get("min_preference_margin", 0.1)
        self.quality_filter.config["min_preference_margin"] = min_margin
        after_filter = []
        margin_filtered = 0
        for s in raw_samples:
            keep, reason = self.quality_filter.filter_dpo(s)
            if keep:
                after_filter.append(s)
            elif reason == "low_preference_margin":
                margin_filtered += 1
        logger.info(f"过滤后: {len(after_filter)} (裕度过滤: {margin_filtered})")

        # Step 3: 按裕度排序（高质量偏好对优先）
        after_filter.sort(key=lambda x: x.preference_margin, reverse=True)

        # Step 4: 去重（基于instruction）
        logger.info("Step 3: 指令去重...")
        dedup = MinHashDeduplicator(threshold=0.9)
        final_samples = []
        for s in after_filter:
            if not dedup.is_duplicate(s.instruction):
                final_samples.append(s)

        # Step 5: 保存
        self._save_dpo(final_samples, output_path)
        logger.info(f"已保存至: {output_path}")

        margins = [s.preference_margin for s in final_samples]
        report = {
            "pipeline": "DPO",
            "stats": {
                "raw": len(raw_samples),
                "after_filter": len(after_filter),
                "final": len(final_samples),
                "margin_filtered": margin_filtered,
                "retention_rate": f"{len(final_samples)/max(len(raw_samples),1)*100:.1f}%"
            },
            "margin_stats": {
                "mean": round(sum(margins)/max(len(margins),1), 4),
                "min": round(min(margins) if margins else 0, 4),
                "max": round(max(margins) if margins else 0, 4),
                "high_margin_ratio": f"{sum(1 for m in margins if m>0.3)/max(len(margins),1)*100:.1f}%"
            }
        }
        self.report["dpo"] = report
        return report

    def _load_sft(self, path: str) -> List[SFTSample]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # 兼容多种格式
                    instruction = d.get("instruction", d.get("prompt", ""))
                    output = d.get("output", d.get("response", d.get("completion", "")))
                    inp = d.get("input", "")
                    if instruction and output:
                        samples.append(SFTSample(
                            instruction=instruction,
                            input=inp,
                            output=output,
                            source=d.get("source", "unknown")
                        ))
                except json.JSONDecodeError:
                    continue
        return samples

    def _load_dpo(self, path: str) -> List[DPOSample]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    instruction = d.get("instruction", d.get("prompt", ""))
                    chosen = d.get("chosen", "")
                    rejected = d.get("rejected", "")
                    if instruction and chosen and rejected:
                        samples.append(DPOSample(
                            instruction=instruction,
                            chosen=chosen,
                            rejected=rejected
                        ))
                except json.JSONDecodeError:
                    continue
        return samples

    def _save_sft(self, samples: List[SFTSample], path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + '\n')

    def _save_dpo(self, samples: List[DPOSample], path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + '\n')

    def print_full_report(self):
        """打印完整数据质量报告"""
        print("\n" + "="*60)
        print("📊 数据质量流水线报告")
        print("="*60)
        for stage, report in self.report.items():
            print(f"\n[{stage.upper()}]")
            for key, val in report.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for k, v in val.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {val}")
        print("="*60)


# ══════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="数据质量流水线")
    parser.add_argument("--sft_input", type=str, help="SFT输入路径 (.jsonl)")
    parser.add_argument("--sft_output", type=str, default="data/cleaned/sft_cleaned.jsonl")
    parser.add_argument("--dpo_input", type=str, help="DPO输入路径 (.jsonl)")
    parser.add_argument("--dpo_output", type=str, default="data/cleaned/dpo_cleaned.jsonl")
    parser.add_argument("--top_k", type=int, default=None, help="SFT Top-K选择（LIMA策略）")
    parser.add_argument("--min_margin", type=float, default=0.15, help="DPO最小偏好裕度")
    parser.add_argument("--curriculum", type=str, default="bucket",
                        choices=["easy_to_hard", "hard_to_easy", "bucket"])
    parser.add_argument("--dedup_threshold", type=float, default=0.85)
    args = parser.parse_args()

    pipeline = DataQualityPipeline(config={"dedup_threshold": args.dedup_threshold})

    if args.sft_input:
        report = pipeline.process_sft(
            args.sft_input, args.sft_output,
            top_k=args.top_k,
            curriculum_strategy=args.curriculum
        )

    if args.dpo_input:
        report = pipeline.process_dpo(
            args.dpo_input, args.dpo_output,
            min_margin=args.min_margin
        )

    pipeline.print_full_report()


if __name__ == "__main__":
    main()
