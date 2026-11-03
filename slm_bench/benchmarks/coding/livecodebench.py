"""
Official LiveCodeBench Benchmark Adapter.
Evaluates code generation against time-stamped contest problems from LeetCode, AtCoder, and Codeforces.
Enforces isolated container-based sandbox execution.
If container sandbox or official dataset dependencies are unavailable,
marks the evaluation as UNAVAILABLE rather than substituting custom approximations.
"""

from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric
from ...execution.sandbox import ContainerSandbox
from ...utils.logger import logger


class LiveCodeBenchBenchmark(BaseBenchmarkTask):
    """Evaluates programming synthesis on official LiveCodeBench problems."""

    def __init__(self):
        super().__init__(
            task_id="livecodebench",
            category="coding",
            description="LiveCodeBench official coding evaluation with contaminated-free problem sets.",
            raw_metric_name="pass@1",
            weight=1.0,
        )
        self.sandbox = ContainerSandbox()

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        seed: int = 42,
    ) -> TaskMetric:
        # Check container sandbox availability
        if not self.sandbox.is_available():
            logger.warning("Docker sandbox is unavailable. Marking LiveCodeBench as UNAVAILABLE.")
            return TaskMetric(
                task_id=self.task_id,
                category=self.category,
                status="UNAVAILABLE",
                raw_score=0.0,
                raw_metric_name="pass@1",
                normalized_score=0.0,
                total_samples=0,
                passed_samples=0,
                failed_samples=0,
                invalid_responses=0,
                ci_lower=0.0,
                ci_upper=0.0,
                dataset_name="LiveCodeBench/code_generation_lite",
                dataset_revision="official",
                split="test",
                sub_metrics={"error": "Docker container sandbox is unavailable. Untrusted code execution is prohibited."},
                question_results=[],
            )

        # In case the upstream HF dataset script is blocked by HF datasets 5.0+:
        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status="UNAVAILABLE",
            raw_score=0.0,
            raw_metric_name="pass@1",
            normalized_score=0.0,
            total_samples=0,
            passed_samples=0,
            failed_samples=0,
            invalid_responses=0,
            ci_lower=0.0,
            ci_upper=0.0,
            dataset_name="LiveCodeBench/code_generation_lite",
            dataset_revision="official",
            split="test",
            sub_metrics={"error": "LiveCodeBench requires upstream dataset migration from loading scripts to parquet."},
            question_results=[],
        )
