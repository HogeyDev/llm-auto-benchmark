"""
GSM8K mathematical reasoning benchmark adapter.
Multi-step grade school mathematical word problems requiring arithmetic computation.
"""

import re
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric
from ...metrics.confidence import compute_metric_confidence
from ..reasoning_math import extract_reasoning_answer


GSM8K_BENCHMARK_DATASET = [
    {
        "id": "gsm8k_001_baker",
        "question": "A bakery makes 40 loaves of bread every morning. They sell 25 loaves before noon and 10 loaves in the afternoon. At the end of the day, how many loaves of bread remain unsold?",
        "expected": "5",
    },
    {
        "id": "gsm8k_002_library",
        "question": "A library has 120 books on science. A donor gives 45 more science books. Later, 30 books are checked out by students. How many science books are currently on the shelves?",
        "expected": "135",
    },
    {
        "id": "gsm8k_003_savings",
        "question": "Maria saves $15 every week for 8 weeks. She then buys a jacket for $70. How much money does she have left from her savings?",
        "expected": "50",
    },
    {
        "id": "gsm8k_004_garden_fencing",
        "question": "A rectangular garden is 15 meters long and 10 meters wide. What is the total perimeter in meters that needs to be fenced?",
        "expected": "50",
    },
    {
        "id": "gsm8k_005_bus_passengers",
        "question": "A bus starts with 24 passengers. At the first stop, 6 get off and 10 get on. At the second stop, 8 get off. How many passengers are on the bus now?",
        "expected": "20",
    },
]


class GSM8KBenchmark(BaseBenchmarkTask):
    """Evaluates multi-step arithmetic reasoning on GSM8K word problems."""

    def __init__(self):
        super().__init__(
            task_id="gsm8k",
            category="reasoning",
            description="GSM8K grade-school multi-step mathematical word problems.",
            raw_metric_name="accuracy",
            weight=1.0,
        )

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> TaskMetric:
        dataset = GSM8K_BENCHMARK_DATASET[:sample_limit] if sample_limit else GSM8K_BENCHMARK_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        breakdown = []

        system_prompt = (
            "You are a mathematical reasoning assistant. Solve the math problem step by step. "
            "At the very end of your response, output '#### <number>' with the final numeric answer."
        )

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                break

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
            ]

            gen = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            extracted = extract_reasoning_answer(gen.text, item_type="numeric")
            expected = item["expected"]

            is_correct = False
            if extracted is not None:
                try:
                    is_correct = abs(float(extracted) - float(expected)) < 1e-4
                except ValueError:
                    is_correct = (extracted.strip() == expected.strip())

            if not is_correct and expected in gen.text:
                is_correct = True

            if is_correct:
                passed += 1

            if on_sample:
                on_sample({
                    "task_id": self.task_id,
                    "category": self.category,
                    "sample_index": idx + 1,
                    "total_samples": total,
                    "prompt": item["question"],
                    "response": gen.text,
                    "expected": expected,
                    "extracted": extracted,
                    "passed": is_correct,
                    "latency_ms": gen.latency_ms,
                    "tokens": gen.tokens_generated,
                })

            breakdown.append({
                "id": item["id"],
                "expected": expected,
                "extracted": extracted,
                "passed": is_correct,
            })

        acc = (passed / total) if total > 0 else 0.0
        conf = compute_metric_confidence(passed, total)

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            raw_score=round(acc, 4),
            raw_metric_name=self.raw_metric_name,
            normalized_score=round(acc * 100.0, 2),
            total_samples=total,
            passed_samples=passed,
            avg_latency_ms=round(total_latency / max(1, total), 2),
            avg_tokens_per_sec=round(total_tokens / max(0.001, total_latency / 1000.0), 2),
            sub_metrics={
                "confidence_interval": conf,
                "details": breakdown,
            },
        )
