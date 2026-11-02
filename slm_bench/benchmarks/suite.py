"""
Benchmark suite orchestrator, score normalization, and composite SLM-Index calculation.
"""

import uuid
from typing import List, Dict, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..backends.base import BaseInferenceBackend
from ..models.schema import BenchmarkResult, TaskMetric, CategoryScore, ModelMetadata, HardwareInfo
from .base import BaseBenchmarkTask
from .reasoning_math import ReasoningBenchmark  # Note: import from module
from .coding_agent import CodingBenchmark
from .instruction_following import InstructionFollowingBenchmark
from .tool_calling import ToolCallingBenchmark
from ..utils.logger import console, logger


DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "reasoning": 0.25,
    "coding": 0.30,
    "instruction": 0.25,
    "tool_calling": 0.20,
}


class BenchmarkSuite:
    """Orchestrates multi-task SLM evaluation and composite score aggregation."""

    def __init__(self, category_weights: Optional[Dict[str, float]] = None):
        self.weights = category_weights or DEFAULT_CATEGORY_WEIGHTS
        self.tasks: Dict[str, BaseBenchmarkTask] = {
            "reasoning": ReasoningBenchmark(),
            "coding": CodingBenchmark(),
            "instruction": InstructionFollowingBenchmark(),
            "tool_calling": ToolCallingBenchmark(),
        }

    def run_suite(
        self,
        backend: BaseInferenceBackend,
        model_meta: ModelMetadata,
        hardware_info: HardwareInfo,
        selected_tasks: Optional[List[str]] = None,
        sample_limit: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Execute evaluation suite and return standardized BenchmarkResult.
        """
        logger.info(f"Starting evaluation suite for [bold green]{model_meta.model_id}[/bold green]...")
        active_tasks = {}
        for key, task_obj in self.tasks.items():
            if selected_tasks is None or key in selected_tasks or task_obj.task_id in selected_tasks:
                active_tasks[key] = task_obj

        task_metrics: Dict[str, TaskMetric] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_bar = progress.add_task("[cyan]Running Benchmarks...", total=len(active_tasks))

            for cat_key, task in active_tasks.items():
                progress.update(main_bar, description=f"[cyan]Evaluating: {task.task_id} ({task.category})")
                metric = task.run(backend, sample_limit=sample_limit)
                task_metrics[task.task_id] = metric
                progress.advance(main_bar)

        # Aggregate by category
        category_scores: Dict[str, CategoryScore] = {}
        for cat_name, cat_weight in self.weights.items():
            matching_metrics = [m for m in task_metrics.values() if m.category == cat_name]
            if matching_metrics:
                avg_norm = sum(m.normalized_score for m in matching_metrics) / len(matching_metrics)
                category_scores[cat_name] = CategoryScore(
                    category=cat_name,
                    weight=cat_weight,
                    normalized_score=round(avg_norm, 2),
                    task_count=len(matching_metrics),
                )
            else:
                category_scores[cat_name] = CategoryScore(
                    category=cat_name,
                    weight=cat_weight,
                    normalized_score=0.0,
                    task_count=0,
                )

        # Calculate SLM-Index composite score
        total_weight = sum(cs.weight for cs in category_scores.values() if cs.task_count > 0) or 1.0
        weighted_sum = sum(cs.normalized_score * cs.weight for cs in category_scores.values() if cs.task_count > 0)
        slm_index = round(weighted_sum / total_weight, 2)

        eval_id = str(uuid.uuid4())
        logger.info(f"Evaluation complete. Aggregate SLM-Index: [bold yellow]{slm_index}%[/bold yellow]")

        return BenchmarkResult(
            schema_version="1.0.0",
            evaluation_id=eval_id,
            model=model_meta,
            hardware=hardware_info,
            tasks=task_metrics,
            categories=category_scores,
            slm_index=slm_index,
            notes=notes,
        )
