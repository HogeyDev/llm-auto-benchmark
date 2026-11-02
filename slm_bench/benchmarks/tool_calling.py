"""
Function calling and multi-turn tool interaction benchmark task.
"""

import json
import re
from typing import Optional, List, Dict, Any
from .base import BaseBenchmarkTask
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a specified location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state/country"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search internal repository database for articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for"},
                },
                "required": ["query"],
            },
        },
    },
]


TOOL_EVAL_SCENARIOS = [
    {
        "scenario_id": "weather_lookup_tokyo",
        "user_query": "What is the weather currently in Tokyo in Celsius?",
        "expected_tool": "get_weather",
        "expected_args_subset": {"location": "Tokyo"},
        "mock_tool_response": {"temperature": 18, "unit": "celsius", "condition": "Sunny"},
        "follow_up_keyword": "18",
    },
    {
        "scenario_id": "math_calculation",
        "user_query": "Can you calculate 25 + 75 using the calculator tool?",
        "expected_tool": "calculator",
        "expected_args_subset": {"operation": "add", "a": 25, "b": 75},
        "mock_tool_response": {"result": 100},
        "follow_up_keyword": "100",
    },
    {
        "scenario_id": "db_search",
        "user_query": "Look up our internal docs for 'model evaluation standards'.",
        "expected_tool": "search_database",
        "expected_args_subset": {"query": "model evaluation standards"},
        "mock_tool_response": {"found": 1, "title": "Standardized Benchmarking Guide"},
        "follow_up_keyword": "Standardized Benchmarking Guide",
    },
]


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from json block or raw string."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            return data
    except json.JSONDecodeError:
        pass
    return None


class ToolCallingBenchmark(BaseBenchmarkTask):
    """Evaluates schema adherence, tool selection, argument parsing, and multi-turn execution."""

    def __init__(self):
        super().__init__(
            task_id="slm_tool_calling_multiturn",
            category="tool_calling",
            description="Multi-turn function selection, argument validation, and tool interaction.",
            raw_metric_name="tool_accuracy",
            weight=1.0,
        )

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
    ) -> TaskMetric:
        scenarios = TOOL_EVAL_SCENARIOS[:sample_limit] if sample_limit else TOOL_EVAL_SCENARIOS
        total = len(scenarios)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        breakdown = []

        system_prompt = (
            "You are an assistant equipped with tools. When a tool is needed, respond ONLY with "
            'a JSON object: {"name": "<tool_name>", "arguments": {<args>}}'
        )

        for sc in scenarios:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Available tools:\n{json.dumps(TOOL_SPECS)}\n\nUser request: {sc['user_query']}"},
            ]

            # Turn 1: Model initiates tool call
            gen1 = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen1.latency_ms
            total_tokens += gen1.tokens_generated

            tool_call = parse_tool_call(gen1.text)
            tool_call_valid = False

            if tool_call:
                name_match = (tool_call.get("name") == sc["expected_tool"])
                args = tool_call.get("arguments", {})
                args_match = all(args.get(k) == v for k, v in sc["expected_args_subset"].items())
                tool_call_valid = name_match and args_match

            # Turn 2: Feed back mock tool response to test multi-turn completion
            messages.append({"role": "assistant", "content": gen1.text})
            messages.append({
                "role": "user",
                "content": f"Tool '{sc['expected_tool']}' output: {json.dumps(sc['mock_tool_response'])}. Please give the final answer.",
            })

            gen2 = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen2.latency_ms
            total_tokens += gen2.tokens_generated

            synthesis_valid = (sc["follow_up_keyword"].lower() in gen2.text.lower())
            scenario_passed = tool_call_valid and synthesis_valid

            if scenario_passed:
                passed += 1

            breakdown.append({
                "scenario_id": sc["scenario_id"],
                "tool_call_valid": tool_call_valid,
                "synthesis_valid": synthesis_valid,
                "passed": scenario_passed,
            })

        acc = (passed / total) if total > 0 else 0.0
        norm_score = acc * 100.0
        avg_lat = (total_latency / (total * 2)) if total > 0 else 0.0
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
            sub_metrics={"scenarios": breakdown},
        )
