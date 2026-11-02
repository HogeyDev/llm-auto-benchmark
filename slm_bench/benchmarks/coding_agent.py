"""
Agentic coding and program synthesis benchmark task with sandbox verification.
Evaluates algorithmic synthesis, edge-case robustness, and practical systems programming.
"""

import re
import datetime
from typing import Optional, List, Dict, Any
from .base import BaseBenchmarkTask
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric
from ..utils.logger import logger


CODING_DATASET = [
    {
        "id": "code_fibonacci",
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
        "id": "code_is_palindrome",
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
        "id": "code_flatten_list",
        "prompt": "Write a Python function `flatten_list(nested: list) -> list` that flattens an arbitrarily nested list of integers into a single flat list.",
        "entry_point": "flatten_list",
        "test_cases": [
            ("flatten_list([1, [2, [3, 4], 5], 6])", [1, 2, 3, 4, 5, 6]),
            ("flatten_list([])", []),
            ("flatten_list([[1, 2], [3]])", [1, 2, 3]),
        ],
    },
    {
        "id": "code_two_sum",
        "prompt": "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns the 0-based indices of the two numbers such that they add up to target.",
        "entry_point": "two_sum",
        "test_cases": [
            ("two_sum([2, 7, 11, 15], 9)", [0, 1]),
            ("two_sum([3, 2, 4], 6)", [1, 2]),
            ("two_sum([3, 3], 6)", [0, 1]),
        ],
    },
    {
        "id": "code_count_vowels",
        "prompt": "Write a Python function `count_vowels(s: str) -> int` that returns the count of vowels (a, e, i, o, u, case-insensitive) in string s.",
        "entry_point": "count_vowels",
        "test_cases": [
            ("count_vowels('hello world')", 3),
            ("count_vowels('AEIOU')", 5),
            ("count_vowels('xyz')", 0),
        ],
    },
    {
        "id": "code_find_longest_word",
        "prompt": "Write a Python function `find_longest_word(sentence: str) -> str` that returns the longest word in a space-separated sentence. If tied, return the first.",
        "entry_point": "find_longest_word",
        "test_cases": [
            ("find_longest_word('The quick brown fox')", "quick"),
            ("find_longest_word('Python programming is awesome')", "programming"),
            ("find_longest_word('one two six')", "one"),
        ],
    },
    {
        "id": "code_is_valid_parentheses",
        "prompt": "Write a Python function `is_valid_parentheses(s: str) -> bool` that returns True if the bracket string containing '()[]{}' has validly nested and matched pairs.",
        "entry_point": "is_valid_parentheses",
        "test_cases": [
            ("is_valid_parentheses('()[]{}')", True),
            ("is_valid_parentheses('(]')", False),
            ("is_valid_parentheses('([)]')", False),
            ("is_valid_parentheses('{[]}')", True),
            ("is_valid_parentheses('')", True),
        ],
    },
    {
        "id": "code_find_missing_number",
        "prompt": "Write a Python function `find_missing_number(nums: list[int]) -> int` that returns the only missing number from the sequence 0 to n in list nums.",
        "entry_point": "find_missing_number",
        "test_cases": [
            ("find_missing_number([3, 0, 1])", 2),
            ("find_missing_number([0, 1])", 2),
            ("find_missing_number([9, 6, 4, 2, 3, 5, 7, 0, 1])", 8),
        ],
    },
    {
        "id": "code_max_subarray_sum",
        "prompt": "Write a Python function `max_subarray_sum(nums: list[int]) -> int` using Kadane's algorithm to return the maximum contiguous subarray sum (handles all-negative lists).",
        "entry_point": "max_subarray_sum",
        "test_cases": [
            ("max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])", 6),
            ("max_subarray_sum([1])", 1),
            ("max_subarray_sum([5, 4, -1, 7, 8])", 23),
            ("max_subarray_sum([-5, -2, -8])", -2),
        ],
    },
    {
        "id": "code_merge_intervals",
        "prompt": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping interval pairs.",
        "entry_point": "merge_intervals",
        "test_cases": [
            ("merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])", [[1, 6], [8, 10], [15, 18]]),
            ("merge_intervals([[1, 4], [4, 5]])", [[1, 5]]),
            ("merge_intervals([])", []),
        ],
    },
    {
        "id": "code_parse_ip_address",
        "prompt": "Write a Python function `parse_ip_address(ip: str) -> bool` that returns True if string `ip` is a valid IPv4 address (4 octets, 0-255, no leading zeroes unless '0').",
        "entry_point": "parse_ip_address",
        "test_cases": [
            ("parse_ip_address('192.168.1.1')", True),
            ("parse_ip_address('256.100.0.1')", False),
            ("parse_ip_address('192.168.01.1')", False),
            ("parse_ip_address('1.2.3')", False),
            ("parse_ip_address('abc.def.ghi.jkl')", False),
        ],
    },
    {
        "id": "code_extract_log_errors",
        "prompt": "Write a Python function `extract_log_errors(logs: str) -> list[str]` that extracts the error message strings from lines in `logs` starting with '[ERROR] '.",
        "entry_point": "extract_log_errors",
        "test_cases": [
            ("extract_log_errors('[INFO] Start\\n[ERROR] Database down\\n[DEBUG] Trace\\n[ERROR] Out of memory')", ["Database down", "Out of memory"]),
            ("extract_log_errors('[INFO] Everything normal')", []),
        ],
    },
    {
        "id": "code_calculate_date_difference",
        "prompt": "Write a Python function `calculate_date_difference(d1: str, d2: str) -> int` that calculates the absolute difference in days between two 'YYYY-MM-DD' date strings.",
        "entry_point": "calculate_date_difference",
        "test_cases": [
            ("calculate_date_difference('2025-01-01', '2025-01-10')", 9),
            ("calculate_date_difference('2025-02-28', '2025-03-01')", 1),
            ("calculate_date_difference('2024-01-01', '2024-01-01')", 0),
        ],
    },
    {
        "id": "code_compress_string",
        "prompt": "Write a Python function `compress_string(s: str) -> str` performing run-length encoding (e.g. 'aabcccccaaa' -> 'a2b1c5a3'). If the compressed string is not shorter, return the original string.",
        "entry_point": "compress_string",
        "test_cases": [
            ("compress_string('aabcccccaaa')", "a2b1c5a3"),
            ("compress_string('abcdef')", "abcdef"),
            ("compress_string('')", ""),
        ],
    },
    {
        "id": "code_longest_common_prefix",
        "prompt": "Write a Python function `longest_common_prefix(strs: list[str]) -> str` that finds the longest common prefix string amongst an array of strings.",
        "entry_point": "longest_common_prefix",
        "test_cases": [
            ("longest_common_prefix(['flower', 'flow', 'flight'])", "fl"),
            ("longest_common_prefix(['dog', 'racecar', 'car'])", ""),
            ("longest_common_prefix(['interstellar', 'internet', 'interval'])", "inte"),
        ],
    },
    {
        "id": "code_is_power_of_two",
        "prompt": "Write a Python function `is_power_of_two(n: int) -> bool` that returns True if integer n is a positive power of two, and False otherwise.",
        "entry_point": "is_power_of_two",
        "test_cases": [
            ("is_power_of_two(1)", True),
            ("is_power_of_two(16)", True),
            ("is_power_of_two(3)", False),
            ("is_power_of_two(0)", False),
            ("is_power_of_two(-16)", False),
        ],
    },
    {
        "id": "code_reverse_words",
        "prompt": "Write a Python function `reverse_words(s: str) -> str` that reverses the order of words in string s, trimming leading/trailing spaces and collapsing multi-spaces to single.",
        "entry_point": "reverse_words",
        "test_cases": [
            ("reverse_words('the sky is blue')", "blue is sky the"),
            ("reverse_words('  hello world  ')", "world hello"),
            ("reverse_words('a good   example')", "example good a"),
        ],
    },
    {
        "id": "code_top_k_frequent",
        "prompt": "Write a Python function `top_k_frequent(nums: list[int], k: int) -> list[int]` that returns the k most frequent elements in nums (sorted in descending order of frequency).",
        "entry_point": "top_k_frequent",
        "test_cases": [
            ("top_k_frequent([1, 1, 1, 2, 2, 3], 2)", [1, 2]),
            ("top_k_frequent([1], 1)", [1]),
        ],
    },
    {
        "id": "code_json_deep_find",
        "prompt": "Write a Python function `json_deep_find(data: dict, target_key: str) -> list` that recursively searches a nested dictionary/list structure and returns all values mapped to target_key.",
        "entry_point": "json_deep_find",
        "test_cases": [
            ("json_deep_find({'a': 1, 'b': {'a': 2, 'c': 3}, 'd': [{'a': 4}]}, 'a')", [1, 2, 4]),
            ("json_deep_find({'x': 10}, 'y')", []),
        ],
    },
    {
        "id": "code_sliding_window_max",
        "prompt": "Write a Python function `sliding_window_max(nums: list[int], k: int) -> list[int]` that returns the maximum element in each sliding window of size k moving from left to right.",
        "entry_point": "sliding_window_max",
        "test_cases": [
            ("sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)", [3, 3, 5, 5, 6, 7]),
            ("sliding_window_max([1], 1)", [1]),
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
    namespace: Dict[str, Any] = {
        "datetime": datetime,
        "re": re,
    }
    try:
        # Precompile and execute definition
        compiled = compile(code_str, "<string>", "exec")
        exec(compiled, namespace)

        if entry_point not in namespace:
            return False, f"Function '{entry_point}' not found in generated code."

        fn = namespace[entry_point]

        for call_expr, expected in test_cases:
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
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
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
            if cancel_event and cancel_event.is_set():
                break
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

            if on_sample:
                on_sample({
                    "task_id": self.task_id,
                    "category": self.category,
                    "sample_index": idx + 1,
                    "total_samples": total,
                    "prompt": item["prompt"],
                    "response": gen.text,
                    "expected": "All unit tests pass",
                    "extracted": "Passed" if success else msg,
                    "passed": success,
                    "latency_ms": gen.latency_ms,
                    "tokens": gen.tokens_generated,
                })

            details.append({
                "index": idx,
                "id": item.get("id", str(idx)),
                "entry_point": item["entry_point"],
                "passed": success,
                "message": msg,
            })

        pass_rate = (passed / total) if total > 0 else 0.0
        norm_score = pass_rate * 100.0
        avg_lat = (total_latency / total) if total > 0 else 0.0
        avg_tps = (total_tokens / (total_latency / 1000.0)) if total_latency > 0 else 0.0

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            raw_score=round(pass_rate, 4),
            raw_metric_name=self.raw_metric_name,
            normalized_score=round(norm_score, 2),
            total_samples=total,
            passed_samples=passed,
            avg_latency_ms=round(avg_lat, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={"sample_details": details},
        )
