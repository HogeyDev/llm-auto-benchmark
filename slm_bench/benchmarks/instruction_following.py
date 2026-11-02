"""
Instruction following and strict constraint compliance benchmark task (IFEval-style).
"""

import json
import re
from typing import Optional, List, Dict, Any, Callable
from .base import BaseBenchmarkTask
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric


def verify_json_schema(text: str, required_keys: List[str]) -> bool:
    """Check if the text contains a valid JSON object with the required keys."""
    # Find json substring if surrounded by markdown
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return False
    try:
        data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return False
        return all(k in data for k in required_keys)
    except json.JSONDecodeError:
        return False


def verify_negative_constraint(text: str, forbidden_substr: str) -> bool:
    """Verify that forbidden substring or character is not present in output."""
    return forbidden_substr.lower() not in text.lower()


def verify_bullet_count(text: str, expected_count: int) -> bool:
    """Count bullet points (lines starting with *, -, or numbers)."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    bullets = [line for line in lines if re.match(r"^(\*|-|\+|\d+\.)\s+", line)]
    return len(bullets) == expected_count


def verify_word_count(text: str, min_words: int, max_words: int) -> bool:
    """Verify text length in word counts."""
    words = text.strip().split()
    return min_words <= len(words) <= max_words


INSTRUCTION_DATASET = [
    {
        "id": "json_france_metadata",
        "prompt": "Provide the country information for France. Respond ONLY with a valid JSON object containing the keys: 'country', 'capital', and 'population_millions'. Do not include markdown or explanations.",
        "validator": lambda t: verify_json_schema(t, ["country", "capital", "population_millions"]),
        "rule": "Strict JSON schema with required keys",
    },
    {
        "id": "negative_constraint_e",
        "prompt": "Write a short sentence about an animal jumping without using the letter 'e' (neither lowercase 'e' nor uppercase 'E').",
        "validator": lambda t: verify_negative_constraint(t, "e") and len(t.strip()) > 5,
        "rule": "Negative constraint: no letter 'e'",
    },
    {
        "id": "exact_three_bullets",
        "prompt": "List 3 primary benefits of automated unit testing. You must format your answer as EXACTLY 3 bullet points starting with '*'. No introductory or concluding text.",
        "validator": lambda t: verify_bullet_count(t, 3),
        "rule": "Format constraint: exactly 3 bullet points",
    },
    {
        "id": "word_count_limit",
        "prompt": "Summarize what an API is in exactly between 5 and 15 words.",
        "validator": lambda t: verify_word_count(t, 5, 15),
        "rule": "Length constraint: between 5 and 15 words",
    },
    {
        "id": "json_status_response",
        "prompt": "Output a valid JSON object with keys 'status' (set to 'success') and 'result'. Nothing else.",
        "validator": lambda t: verify_json_schema(t, ["status", "result"]),
        "rule": "Strict JSON format",
    },
]


class InstructionFollowingBenchmark(BaseBenchmarkTask):
    """Evaluates strict instruction following and structural constraint obedience."""

    def __init__(self):
        super().__init__(
            task_id="slm_instruction_following",
            category="instruction",
            description="IFEval-style strict constraint compliance, JSON schema obedience, and formatting.",
            raw_metric_name="compliance_rate",
            weight=1.0,
        )

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
    ) -> TaskMetric:
        dataset = INSTRUCTION_DATASET[:sample_limit] if sample_limit else INSTRUCTION_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        breakdown = []

        system_prompt = "You are an instruction-following assistant. You must adhere strictly to all user constraints."

        for idx, item in enumerate(dataset):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
            ]
            gen = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            is_compliant = False
            try:
                is_compliant = bool(item["validator"](gen.text))
            except Exception:
                is_compliant = False

            if is_compliant:
                passed += 1

            breakdown.append({
                "id": item["id"],
                "rule": item["rule"],
                "compliant": is_compliant,
                "output_preview": gen.text[:100],
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
            sub_metrics={"constraints": breakdown},
        )
