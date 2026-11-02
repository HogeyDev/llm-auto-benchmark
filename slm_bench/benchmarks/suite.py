"""
Benchmark suite orchestrator, score normalization, and composite SLM-Index calculation.
"""

import uuid
from typing import List, Dict, Optional, Any, Callable
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..backends.base import BaseInferenceBackend
from ..execution.inference_gateway import InferenceGateway
from ..metrics.telemetry import PerformanceCollector
from ..metrics.confidence import wilson_score_interval
from ..scoring.scoring_engine import ScoringEngine, DEFAULT_CATEGORY_WEIGHTS
from ..models.schema import BenchmarkResult, TaskMetric, CategoryScore, ModelMetadata, HardwareInfo
from .base import BaseBenchmarkTask
from .registry import get_task, get_profile_tasks, BENCHMARK_REGISTRY
from ..utils.logger import console, logger


class BenchmarkSuite:
    """Orchestrates multi-task SLM evaluation, telemetry, and composite index aggregation."""

    def __init__(self, category_weights: Optional[Dict[str, float]] = None):
        self.weights = category_weights or DEFAULT_CATEGORY_WEIGHTS
        self.scoring_engine = ScoringEngine(category_weights=self.weights)
        # Default core tasks for backwards compatibility
        self.tasks: Dict[str, BaseBenchmarkTask] = {
            "reasoning": get_task("slm_reasoning_cot"),
            "coding": get_task("slm_coding_agent"),
            "instruction": get_task("slm_instruction_ifeval"),
            "tool_calling": get_task("slm_tool_calling_multiturn"),
        }

    def resolve_tasks(
        self,
        selected_tasks: Optional[List[str]] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, BaseBenchmarkTask]:
        """Resolve active benchmark task instances based on profile or explicit task list."""
        if selected_tasks:
            resolved: Dict[str, BaseBenchmarkTask] = {}
            for name in selected_tasks:
                clean_name = name.strip()
                if clean_name in self.tasks:
                    resolved[self.tasks[clean_name].task_id] = self.tasks[clean_name]
                else:
                    try:
                        task_obj = get_task(clean_name)
                        resolved[task_obj.task_id] = task_obj
                    except ValueError:
                        logger.warning(f"Skipping unrecognized task: '{clean_name}'")
            return resolved

        if profile:
            profile_ids = get_profile_tasks(profile)
            resolved = {}
            for tid in profile_ids:
                try:
                    task_obj = get_task(tid)
                    resolved[task_obj.task_id] = task_obj
                except Exception as e:
                    logger.warning(f"Could not load task '{tid}' for profile '{profile}': {e}")
            return resolved

        # Default fallback to 4 core tasks
        return {t.task_id: t for t in self.tasks.values()}

    def run_suite(
        self,
        backend: BaseInferenceBackend,
        model_meta: ModelMetadata,
        hardware_info: HardwareInfo,
        selected_tasks: Optional[List[str]] = None,
        profile: Optional[str] = "standard",
        sample_limit: Optional[int] = None,
        notes: Optional[str] = None,
        on_task_start: Optional[Any] = None,
        on_sample: Optional[Any] = None,
        on_task_complete: Optional[Any] = None,
        show_progress: bool = True,
        cancel_event: Optional[Any] = None,
    ) -> BenchmarkResult:
        """
        Execute evaluation suite and return standardized BenchmarkResult with multi-track indices.
        """
        logger.info(f"Starting evaluation suite for [bold green]{model_meta.model_id}[/bold green] (profile: {profile or 'custom'})...")
        active_tasks = self.resolve_tasks(selected_tasks=selected_tasks, profile=profile)

        # Wrap backend in InferenceGateway with PerformanceCollector
        collector = PerformanceCollector()
        gateway = InferenceGateway(backend=backend, collector=collector)

        task_metrics: Dict[str, TaskMetric] = {}

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                main_bar = progress.add_task("[cyan]Running Benchmarks...", total=len(active_tasks))

                for task_id, task in active_tasks.items():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Benchmark cancelled by user.")
                    if on_task_start:
                        on_task_start(task.task_id, task.category)
                    progress.update(main_bar, description=f"[cyan]Evaluating: {task.task_id} ({task.category})")

                    metric = task.run(gateway.backend, sample_limit=sample_limit, on_sample=on_sample, cancel_event=cancel_event)

                    # Ensure confidence interval is calculated
                    if metric.ci_lower is None or metric.ci_upper is None:
                        ci_l, ci_u = wilson_score_interval(metric.passed_samples, metric.total_samples)
                        metric.ci_lower = ci_l
                        metric.ci_upper = ci_u

                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Benchmark cancelled by user.")
                    task_metrics[task.task_id] = metric
                    if on_task_complete:
                        on_task_complete(task.task_id, metric)
                    progress.advance(main_bar)
        else:
            for task_id, task in active_tasks.items():
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Benchmark cancelled by user.")
                if on_task_start:
                    on_task_start(task.task_id, task.category)

                metric = task.run(gateway.backend, sample_limit=sample_limit, on_sample=on_sample, cancel_event=cancel_event)
                if metric.ci_lower is None or metric.ci_upper is None:
                    ci_l, ci_u = wilson_score_interval(metric.passed_samples, metric.total_samples)
                    metric.ci_lower = ci_l
                    metric.ci_upper = ci_u

                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Benchmark cancelled by user.")
                task_metrics[task.task_id] = metric
                if on_task_complete:
                    on_task_complete(task.task_id, metric)

        # 1. Scoring & Categorization
        category_scores = self.scoring_engine.aggregate_categories(task_metrics)
        slm_index = self.scoring_engine.compute_slm_index(category_scores)
        coding_agentic_index = self.scoring_engine.compute_coding_agentic_index(category_scores)

        # 2. Performance Telemetry & Efficiency Index
        telemetry_summary = collector.get_aggregate_summary()
        avg_tps = telemetry_summary.get("avg_decode_tokens_per_sec")
        if not avg_tps and task_metrics:
            valid_tps = [m.avg_tokens_per_sec for m in task_metrics.values() if m.avg_tokens_per_sec]
            avg_tps = (sum(valid_tps) / len(valid_tps)) if valid_tps else 0.0

        local_efficiency = self.scoring_engine.compute_local_efficiency_index(
            slm_index=slm_index,
            avg_tokens_per_sec=avg_tps,
            model_ram_gb=model_meta.model_ram_gb or hardware_info.model_ram_gb,
            peak_vram_gb=telemetry_summary.get("peak_vram_mb", 0.0) / 1024.0 if telemetry_summary.get("peak_vram_mb") else None,
        )

        eval_id = str(uuid.uuid4())
        logger.info(f"Evaluation complete. Aggregate SLM-Index: [bold yellow]{slm_index}%[/bold yellow] (Coding/Agentic: [cyan]{coding_agentic_index}%[/cyan])")

        return BenchmarkResult(
            schema_version="1.0.0",
            evaluation_id=eval_id,
            model=model_meta,
            hardware=hardware_info,
            tasks=task_metrics,
            categories=category_scores,
            slm_index=slm_index,
            notes=notes,
            profile=profile,
            coding_agentic_index=coding_agentic_index,
            local_efficiency=local_efficiency,
            provenance={
                "profile": profile,
                "benchmark_count": len(task_metrics),
                "evaluator_version": "0.2.0",
                "telemetry": telemetry_summary,
            },
        )
