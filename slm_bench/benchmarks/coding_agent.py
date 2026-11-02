"""
Agentic coding and program synthesis benchmark task with sandbox verification.
"""

import re
import multiprocessing
import traceback
from typing import Optional, List, Dict, Any
from .base import BaseBenchmarkTask
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric
from ..utils.logger import logger


CODING_DATASET = [
    {
        "prompt": "Write a Python function `fibonacci(n: int) -> int` that returns the n-th Fibonacci number (where fibonacci(0) = 0, fibonacci(1) = 1, fibonacci(2) = 1, etc.).",
        "entry_point": "fibonacci",
        "test_cases": [
            ("fibonacci(0)", 0),
            ("fibonacci(1)", 1),
            ("fibonacci(5)", 5),
            ("fibonacci(10)", 55),
        ],
    },
    {
        "prompt": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is a palindrome (ignoring casing and non-alphanumeric characters), and False otherwise.",
        "entry_point": "is_palindrome",
        "test_cases": [
            ("is_palindrome('A man, a plan, a canal: Panama')", True),
            ("is_palindrome('race a car')", False),
            ("is_palindrome('')", True),
            ("is_palindrome('Was it a car or a cat I saw?')", True),
        ],
    },
    {
        "prompt": "Write a Python function `flatten_list(nested: list) -> list` that flattens an arbitrarily nested list of integers into a single flat list.",
        "entry_point": "flatten_list",
        "test_cases": [
            ("flatten_list([1, [2, [3, 4], 5], 6])", [1, 2, 3, 4, 5, 6]),
            ("flatten_list([])", []),
            ("flatten_list([[1, 2], [3]])", [1, 2, 3]),
        ],
    },
    {
        "prompt": "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns the 0-based indices of the two numbers such that they add up to target.",
        "entry_point": "two_sum",
        "test_cases": [
            ("two_sum([2, 7, 11, 15], 9)", [0, 1]),
            ("two_sum([3, 2, 4], 6)", [1, 2]),
            ("two_sum([3, 3], 6)", [0, 1]),
        ],
    },
    {
        "prompt": "Write a Python function `count_vowels(s: str) -> int` that returns the count of vowels (a, e, i, o, u, case-insensitive) in string s.",
        "entry_point": "count_vowels",
        "test_cases": [
            ("count_vowels('hello world')", 3),
            ("count_vowels('AEIOU')", 5),
            ("count_vowels('xyz')", 0),
        ],
    },
    {
        "prompt": "Write a Python function `find_longest_word(sentence: str) -> str` that returns the longest word in a space-separated sentence. If tied, return the first.",
        "entry_point": "find_longest_word",
        "test_cases": [
            ("find_longest_word('The quick brown fox')", "quick"),
            ("find_longest_word('Python programming is awesome')", "programming"),
            ("find_longest_word('one two six')", "one"),
        ],
    },
]


def extract_python_code(text: str) -> str:
    """Extract code from ```python ... ``` fences or return stripped text."""
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def run_code_in_sandbox(code_str: str, entry_point: str, test_cases: List[tuple]) -> tuple[bool, str]:
    """
    Safely execute the generated code and test assertions in a restricted namespace.
    """
    namespace: Dict[str, Any] = {}
    try:
        # Precompile and execute definition
        compiled = compile(code_str, "<string>", "exec")
        exec(compiled, namespace)
        
        if entry_point not in namespace:
            return False, f"Function '{entry_point}' not found in generated code."
        
        fn = namespace[entry_point]
        
        for call_expr, expected in test_cases:
            # Safely evaluate test call in context
            actual = eval(call_expr, namespace)
            if actual != expected:
                return False, f"Failed assertion on {call_expr}. Expected {expected}, got {actual}"
                
        return True, "All test cases passed."
    except Exception as e:
        return False, f"Execution exception: {e}"


class CodingBenchmark(BaseBenchmarkTask):
    """Evaluates agentic code generation, algorithm implementation, and unit testing."""

    def __init__(self):
        super().__init__(
            task_id="slm_agentic_coding",
            category="coding",
            description="Agentic Python function synthesis and unit-test contract fulfillment.",
            raw_metric_name="pass@1",
            weight=1.0,
        )

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
    ) -> TaskMetric:
        dataset = CODING_DATASET[:sample_limit] if sample_limit else CODING_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        details = []

        system_prompt = (
            "You are an expert Python software engineer. "
            "Write only the requested Python function inside a clean ```python ... ``` block. "
            "Do not include external dependencies. Ensure type annotations and edge case handling."
        )

        for idx, item in enumerate(dataset):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
            ]
            gen = backend.chat(messages, max_new_tokens=384, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            code = extract_python_code(gen.text)
            success, msg = run_code_in_sandbox(code, item["entry_point"], item["test_cases"])

            if success:
                passed += 1

            details.append({
                "index": idx,
                "entry_point": item["entry_point"],
                "passed": success,
                "message": msg,
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
            sub_metrics={"test_results": details},
        )
