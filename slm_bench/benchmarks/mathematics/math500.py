"""
MATH-500 competition mathematics benchmark adapter.
Challenging competition-level problems across algebra, number theory, geometry, and probability.
"""

import re
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric
from ...metrics.confidence import compute_metric_confidence
from ..reasoning_math import extract_reasoning_answer


MATH500_BENCHMARK_DATASET = [
    {
        "id": "math500_algebra_quadratics",
        "subject": "algebra",
        "question": "What is the positive difference between the roots of the quadratic equation x^2 - 14x + 40 = 0?",
        "expected": "6",
    },
    {
        "id": "math500_number_theory_primes",
        "subject": "number_theory",
        "question": "How many positive integers less than 30 are relatively prime to 30?",
        "expected": "8",
    },
    {
        "id": "math500_geometry_pythagoras",
        "subject": "geometry",
        "question": "A right triangle has hypotenuse 13 and one leg of length 5. What is the area of the triangle?",
        "expected": "30",
    },
    {
        "id": "math500_probability_dice",
        "subject": "counting_and_probability",
        "question": "Two standard fair 6-sided dice are rolled. What is the probability that the sum of the dice is 7? Express as a fraction or decimal.",
        "expected": "1/6",
    },
]


class MATH500Benchmark(BaseBenchmarkTask):
    """Evaluates competition-level mathematical problem solving on MATH-500."""

    def __init__(self):
        super().__init__(
            task_id="math500",
            category="reasoning",
            description="MATH-500 competition mathematics across algebra, geometry, and number theory.",
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
        dataset = MATH500_BENCHMARK_DATASET[:sample_limit] if sample_limit else MATH500_BENCHMARK_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        breakdown = []

        system_prompt = (
            "You are a competition mathematician. Work through the derivation clearly. "
            "At the very end of your response, output your final result in the format '#### <answer>' or '\\boxed{<answer>}'."
        )

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                break

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Solve this problem: {item['question']}"},
            ]

            gen = backend.chat(messages, max_new_tokens=384, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            extracted = extract_reasoning_answer(gen.text, item_type="numeric")
            expected = item["expected"]

            is_correct = False
            if extracted is not None:
                if extracted.strip() == expected.strip():
                    is_correct = True
                else:
                    try:
                        is_correct = abs(float(extracted) - float(eval(expected))) < 1e-4
                    except Exception:
                        pass

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
                "subject": item.get("subject"),
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
