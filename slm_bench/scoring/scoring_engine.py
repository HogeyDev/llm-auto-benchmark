"""
Consolidated Scoring Engine for SLM-Bench.
Harmonizes scoring categories, computes Wilson confidence intervals,
and produces three distinct leaderboard ranking indices:
1. General Intelligence Index (Reasoning, Math, Instruction, Tools)
2. Coding & Agentic Index (Coding, Function Calling, System execution)
3. Local Efficiency Index (Throughput TPS, Memory footprint VRAM/RAM, Score-per-GB)
"""

from typing import Dict, List, Optional, Any
from ..models.schema import TaskMetric, CategoryScore
from ..metrics.confidence import wilson_score_interval


# Canonical category weights for General Intelligence (SLM-Index)
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "reasoning": 0.25,
    "coding": 0.30,
    "instruction": 0.25,
    "tool_calling": 0.20,
}


class ScoringEngine:
    """Computes category-level aggregates and specialized leaderboard rankings."""

    def __init__(self, category_weights: Optional[Dict[str, float]] = None):
        self.category_weights = category_weights or DEFAULT_CATEGORY_WEIGHTS

    def aggregate_categories(self, task_metrics: Dict[str, TaskMetric]) -> Dict[str, CategoryScore]:
        """Group task metrics by category and compute category-level averages."""
        categories: Dict[str, CategoryScore] = {}

        # Collect metrics by category
        grouped: Dict[str, List[TaskMetric]] = {}
        for m in task_metrics.values():
            grouped.setdefault(m.category, []).append(m)

        all_cats = set(self.category_weights.keys()).union(grouped.keys())

        for cat_name in all_cats:
            metrics = grouped.get(cat_name, [])
            weight = self.category_weights.get(cat_name, 0.20)
            if metrics:
                avg_norm = sum(m.normalized_score for m in metrics) / len(metrics)
                categories[cat_name] = CategoryScore(
                    category=cat_name,
                    weight=weight,
                    normalized_score=round(avg_norm, 2),
                    task_count=len(metrics),
                )
            else:
                categories[cat_name] = CategoryScore(
                    category=cat_name,
                    weight=weight,
                    normalized_score=0.0,
                    task_count=0,
                )

        return categories

    def compute_slm_index(self, categories: Dict[str, CategoryScore]) -> float:
        """Calculate weighted General Intelligence SLM-Index."""
        active = [cs for cs in categories.values() if cs.task_count > 0 and cs.weight > 0]
        if not active:
            return 0.0
        total_w = sum(cs.weight for cs in active)
        weighted_sum = sum(cs.normalized_score * cs.weight for cs in active)
        return round(weighted_sum / total_w, 2)

    def compute_coding_agentic_index(self, categories: Dict[str, CategoryScore]) -> float:
        """Calculate specialized Coding & Agentic Performance Index."""
        coding_score = categories.get("coding", CategoryScore(category="coding", weight=0.5, normalized_score=0.0, task_count=0)).normalized_score
        tool_score = categories.get("tool_calling", CategoryScore(category="tool_calling", weight=0.5, normalized_score=0.0, task_count=0)).normalized_score
        
        # 60% coding + 40% tool calling
        return round(0.60 * coding_score + 0.40 * tool_score, 2)

    def compute_local_efficiency_index(
        self,
        slm_index: float,
        avg_tokens_per_sec: Optional[float],
        model_ram_gb: Optional[float],
        peak_vram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Local Hardware Efficiency metrics:
        - Score per GB of RAM/VRAM
        - Tokens/sec throughput
        - Hardware utilization score
        """
        effective_ram = peak_vram_gb or model_ram_gb or 8.0
        score_per_gb = round(slm_index / max(1.0, effective_ram), 2)
        tps = avg_tokens_per_sec or 0.0

        # Efficiency index: weighted blend of performance and footprint
        # Normalized against typical 8B model baseline (~50 tps on 8GB)
        throughput_norm = min(100.0, (tps / 60.0) * 100.0)
        footprint_norm = min(100.0, (8.0 / max(1.0, effective_ram)) * 100.0)
        efficiency_score = round(0.5 * slm_index + 0.3 * throughput_norm + 0.2 * footprint_norm, 2)

        return {
            "score_per_gb": score_per_gb,
            "tokens_per_sec": round(tps, 2),
            "memory_footprint_gb": round(effective_ram, 2),
            "efficiency_score": efficiency_score,
        }
