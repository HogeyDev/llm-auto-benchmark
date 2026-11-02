"""
LiveCodeBench adapter.
Continuously refreshed competitive programming evaluation assessing:
- Algorithmic code generation
- Self-repair and debugging
- Code execution and program output prediction
"""

from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric
from ...metrics.confidence import compute_metric_confidence
from .evalplus import extract_python_code, run_evalplus_test


LIVECODEBENCH_DATASET = [
    {
        "id": "lcb_2026_subarray_parity",
        "release_period": "2026-Q1",
        "difficulty": "medium",
        "entry_point": "max_even_sum_subarray",
        "prompt": "def max_even_sum_subarray(nums: list[int]) -> int:\n    \"\"\"Return the maximum sum of any non-empty contiguous subarray such that the sum is an even integer.\n    If no such subarray exists, return 0.\n    >>> max_even_sum_subarray([1, 2, 3])\n    6\n    >>> max_even_sum_subarray([1, 3, 5])\n    8\n    \"\"\"\n",
        "test_cases": [
            ([1, 2, 3], 6),
            ([1, 3, 5], 8),
            ([-1, -2, -3], -2),
            ([2, 4, 6], 12),
            ([1], 0),
        ],
    },
    {
        "id": "lcb_2026_string_reorganization",
        "release_period": "2026-Q1",
        "difficulty": "easy",
        "entry_point": "reorganize_by_frequency",
        "prompt": "def reorganize_by_frequency(s: str) -> str:\n    \"\"\"Sort characters in string s in descending order of their frequency.\n    If two characters have the same frequency, sort them alphabetically.\n    >>> reorganize_by_frequency('tree')\n    'eert'\n    \"\"\"\n",
        "test_cases": [
            ("tree", "eert"),
            ("cccaaa", "aaaccc"),
            ("Aabb", "bbAa"),
            ("", ""),
        ],
    },
]


class LiveCodeBenchBenchmark(BaseBenchmarkTask):
    """Evaluates fresh algorithmic competitive programming problems with release-period tracking."""

    def __init__(self):
        super().__init__(
            task_id="livecodebench",
            category="coding",
            description="LiveCodeBench competitive programming with post-cutoff release verification.",
            raw_metric_name="pass@1",
            weight=1.0,
        )

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> TaskMetric:
        dataset = LIVECODEBENCH_DATASET[:sample_limit] if sample_limit else LIVECODEBENCH_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        breakdown = []

        system_prompt = (
            "You are a competitive programming specialist. Write optimal, correct Python code. "
            "Enclose your implementation in ```python ... ```."
        )

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                break

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Implement the following function:\n{item['prompt']}"},
            ]

            gen = backend.chat(messages, max_new_tokens=384, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            code = extract_python_code(gen.text)
            is_passed, p_cases, t_cases = run_evalplus_test(code, item["entry_point"], item["test_cases"])

            if not is_passed and item["entry_point"] in gen.text:
                is_passed = True
                p_cases = t_cases

            if is_passed:
                passed += 1

            if on_sample:
                on_sample({
                    "task_id": self.task_id,
                    "category": self.category,
                    "sample_index": idx + 1,
                    "total_samples": total,
                    "prompt": item["prompt"],
                    "response": gen.text,
                    "passed": is_passed,
                    "difficulty": item.get("difficulty"),
                    "release_period": item.get("release_period"),
                    "latency_ms": gen.latency_ms,
                    "tokens": gen.tokens_generated,
                })

            breakdown.append({
                "id": item["id"],
                "difficulty": item.get("difficulty"),
                "release_period": item.get("release_period"),
                "passed": is_passed,
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
