"""
Multi-step mathematical reasoning benchmark task (GSM8K and ARC-style problems).
"""

import re
from typing import Optional, List, Dict, Any
from .base import BaseBenchmarkTask
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric
from ..utils.logger import logger


REASONING_DATASET = [
    {
        "question": "A baker makes 12 cakes per hour. He bakes for 4 hours. Later, he sells 35 cakes to a wedding party. How many cakes does he have left?",
        "expected": "13",
    },
    {
        "question": "A train travels 150 km in 2.5 hours at a constant speed. What is the speed of the train in km/h?",
        "expected": "60",
    },
    {
        "question": "What is the sum of all prime numbers strictly less than 10?",
        "expected": "17",
    },
    {
        "question": "A rectangle has a length of 10 cm and a width of 5 cm. What is its perimeter in cm?",
        "expected": "30",
    },
    {
        "question": "Elena had $50. She bought 3 books costing $12 each. How much money does she have remaining?",
        "expected": "14",
    },
    {
        "question": "If 4 workers can build a wall in 6 days, how many days would it take 8 workers working at the same pace to build the same wall?",
        "expected": "3",
    },
    {
        "question": "A box contains 5 red balls, 3 green balls, and 2 blue balls. If one ball is picked at random, what is the probability (in percentage, 0-100) of picking a green ball?",
        "expected": "30",
    },
    {
        "question": "A store discounts a $120 jacket by 25%. What is the sale price of the jacket?",
        "expected": "90",
    },
]


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract numeric answer from CoT text or '#### <answer>' syntax."""
    # Look for standard GSM8k delimiter
    if "####" in text:
        ans_part = text.split("####")[-1].strip()
        num_match = re.search(r"[-+]?\d+(?:\.\d+)?", ans_part)
        if num_match:
            return num_match.group(0).rstrip(".")
    
    # Check for phrases like "the answer is X"
    match = re.findall(r"(?:answer is|is equal to|leaves|speed =|remaining:?|total:?|sale price:?|\$)\s*([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match[-1].rstrip(".")

    # Fallback to the very last number mentioned in the text
    all_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if all_nums:
        return all_nums[-1].rstrip(".")

    return None


class ReasoningBenchmark(BaseBenchmarkTask):
    """Evaluates multi-step mathematical and symbolic reasoning."""

    def __init__(self):
        super().__init__(
            task_id="slm_reasoning_cot",
            category="reasoning",
            description="Multi-step chain-of-thought mathematical problem solving and reasoning.",
            raw_metric_name="accuracy",
            weight=1.0,
        )

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
    ) -> TaskMetric:
        dataset = REASONING_DATASET[:sample_limit] if sample_limit else REASONING_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        results_breakdown = []

        system_prompt = (
            "You are a helpful mathematical reasoning assistant. "
            "Think step by step carefully. At the very end of your response, write '#### <number>' "
            "with the final numeric answer."
        )

        for idx, item in enumerate(dataset):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Solve this problem: {item['question']}"},
            ]
            gen = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            extracted = extract_numeric_answer(gen.text)
            expected = item["expected"]

            # Compare floats if numeric
            is_correct = False
            if extracted is not None:
                try:
                    is_correct = abs(float(extracted) - float(expected)) < 1e-4
                except ValueError:
                    is_correct = (extracted.strip() == expected.strip())

            if is_correct:
                passed += 1

            results_breakdown.append({
                "index": idx,
                "expected": expected,
                "extracted": extracted,
                "passed": is_correct,
            })

        acc = (passed / total) if total > 0 else 0.0
        norm_score = acc * 100.0
        avg_lat = (total_latency / total) if total > 0 else 0.0
        avg_tps = (total_tokens / (total_latency / 1000.0)) if total_latency > 0 else 0.0

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            raw_score=acc,
            raw_metric_name=self.raw_metric_name,
            normalized_score=round(norm_score, 2),
            total_samples=total,
            passed_samples=passed,
            avg_latency_ms=round(avg_lat, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={"sample_details": results_breakdown},
        )
