"""
EleutherAI LM Evaluation Harness (lm-eval) adapter.
Integrates standard lm-evaluation-harness benchmarks (MMLU-Pro, GPQA, BBH, GSM8K, MATH, IFEval)
with automatic fallback to curated native adapters when lm-eval is not installed.
"""

from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric
from ...utils.logger import logger

try:
    import lm_eval
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False


class LMEvalHarnessAdapter(BaseBenchmarkTask):
    """
    Adapter bridging SLM-Bench inference backends to EleutherAI's lm-evaluation-harness.
    """

    def __init__(self, task_name: str = "gsm8k", category: str = "reasoning", weight: float = 1.0):
        super().__init__(
            task_id=f"lm_eval_{task_name}",
            category=category,
            description=f"LM-Eval Harness standardized evaluation of {task_name}.",
            raw_metric_name="accuracy",
            weight=weight,
        )
        self.task_name = task_name

    def is_available(self) -> bool:
        """Check if lm_eval package is installed."""
        return HAS_LM_EVAL

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> TaskMetric:
        if not HAS_LM_EVAL:
            logger.info(f"lm_eval not installed. Delegating {self.task_name} to native benchmark runner.")
            # Delegate to corresponding native task
            if self.task_name == "gsm8k":
                from ..mathematics.gsm8k import GSM8KBenchmark
                return GSM8KBenchmark().run(backend, sample_limit=sample_limit, on_sample=on_sample, cancel_event=cancel_event)
            elif self.task_name in ("mmlu_pro", "mmlu"):
                from ..reasoning.mmlu_pro import MMLUProBenchmark
                return MMLUProBenchmark().run(backend, sample_limit=sample_limit, on_sample=on_sample, cancel_event=cancel_event)
            elif self.task_name in ("gpqa", "gpqa_diamond"):
                from ..reasoning.gpqa import GPQADiamondBenchmark
                return GPQADiamondBenchmark().run(backend, sample_limit=sample_limit, on_sample=on_sample, cancel_event=cancel_event)
            else:
                from ..reasoning_math import ReasoningBenchmark
                return ReasoningBenchmark().run(backend, sample_limit=sample_limit, on_sample=on_sample, cancel_event=cancel_event)

        # When lm_eval is available, run lm_eval.evaluator
        try:
            from lm_eval import evaluator
            # In real environment, evaluate through lm-eval API
            results = evaluator.simple_evaluate(
                model="hf" if "transformers" in backend.__class__.__name__.lower() else "local",
                tasks=[self.task_name],
                limit=sample_limit,
            )
            raw_acc = results["results"][self.task_name].get("acc,none", 0.0)
            return TaskMetric(
                task_id=self.task_id,
                category=self.category,
                raw_score=round(raw_acc, 4),
                raw_metric_name=self.raw_metric_name,
                normalized_score=round(raw_acc * 100.0, 2),
                total_samples=sample_limit or 100,
                passed_samples=int(raw_acc * (sample_limit or 100)),
                sub_metrics={"lm_eval_raw": results["results"]},
            )
        except Exception as e:
            logger.warning(f"Error executing lm_eval: {e}. Falling back to native runner.")
            from ..reasoning_math import ReasoningBenchmark
            return ReasoningBenchmark().run(backend, sample_limit=sample_limit, on_sample=on_sample, cancel_event=cancel_event)
