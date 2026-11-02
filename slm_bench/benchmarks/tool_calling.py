"""
Function calling and multi-turn tool interaction benchmark task.
Evaluates:
- Tool selection from an 8-tool catalog
- Strict parameter and type validation
- Irrelevance / negative testing (knowing when NOT to invoke a tool)
- Multi-turn execution and tool response synthesis
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
                    "limit": {"type": "integer", "description": "Maximum number of results"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text content from a local file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Absolute or relative file path"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute a read-only SQL query against the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query string"},
                    "database": {"type": "string", "description": "Target database name"},
                },
                "required": ["query", "database"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Email address of recipient"},
                    "subject": {"type": "string", "description": "Subject line"},
                    "body": {"type": "string", "description": "Body message text"},
                },
                "required": ["recipient", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Retrieve current stock price and trading volume for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, NVDA)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Schedule a new event in the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title or meeting name"},
                    "start_time": {"type": "string", "description": "ISO-8601 timestamp (e.g. 2025-06-01T10:00:00)"},
                    "duration_minutes": {"type": "integer", "description": "Meeting length in minutes"},
                },
                "required": ["title", "start_time"],
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
        "scenario_id": "math_calculation_add",
        "user_query": "Can you calculate 25 + 75 using the calculator tool?",
        "expected_tool": "calculator",
        "expected_args_subset": {"operation": "add", "a": 25, "b": 75},
        "mock_tool_response": {"result": 100},
        "follow_up_keyword": "100",
    },
    {
        "scenario_id": "db_search_eval_standards",
        "user_query": "Look up our internal docs for 'model evaluation standards'.",
        "expected_tool": "search_database",
        "expected_args_subset": {"query": "model evaluation standards"},
        "mock_tool_response": {"found": 1, "title": "Standardized Benchmarking Guide"},
        "follow_up_keyword": "Standardized Benchmarking Guide",
    },
    {
        "scenario_id": "read_nginx_config",
        "user_query": "Read the configuration file located at '/etc/nginx/nginx.conf'.",
        "expected_tool": "read_file",
        "expected_args_subset": {"filepath": "/etc/nginx/nginx.conf"},
        "mock_tool_response": {"content": "events { worker_connections 1024; }"},
        "follow_up_keyword": "worker_connections",
    },
    {
        "scenario_id": "sql_active_users",
        "user_query": "Execute an SQL query to select all active users from the 'prod_db' database.",
        "expected_tool": "execute_sql",
        "expected_args_subset": {"database": "prod_db"},
        "mock_tool_response": {"rows": [{"id": 1, "username": "alice"}, {"id": 2, "username": "bob"}]},
        "follow_up_keyword": "alice",
    },
    {
        "scenario_id": "send_alert_email",
        "user_query": "Send an email to devops@company.com with subject 'Disk Alert' and body 'Disk space is above 90%'.",
        "expected_tool": "send_email",
        "expected_args_subset": {"recipient": "devops@company.com", "subject": "Disk Alert"},
        "mock_tool_response": {"status": "sent", "id": "msg_9843"},
        "follow_up_keyword": "sent",
    },
    {
        "scenario_id": "stock_quote_nvda",
        "user_query": "What is the current stock price of NVDA?",
        "expected_tool": "get_stock_quote",
        "expected_args_subset": {"ticker": "NVDA"},
        "mock_tool_response": {"ticker": "NVDA", "price": 128.50, "volume": 42000000},
        "follow_up_keyword": "128.5",
    },
    {
        "scenario_id": "schedule_calendar_meeting",
        "user_query": "Schedule a meeting titled 'Architecture Review' starting at '2025-06-01T14:00:00' for 45 minutes.",
        "expected_tool": "create_calendar_event",
        "expected_args_subset": {"title": "Architecture Review", "start_time": "2025-06-01T14:00:00"},
        "mock_tool_response": {"status": "confirmed", "event_id": "cal_301"},
        "follow_up_keyword": "confirmed",
    },
    {
        "scenario_id": "math_calculation_multiply",
        "user_query": "Multiply 12 by 12 using the calculator tool.",
        "expected_tool": "calculator",
        "expected_args_subset": {"operation": "multiply", "a": 12, "b": 12},
        "mock_tool_response": {"result": 144},
        "follow_up_keyword": "144",
    },
    # ─── Negative Scenarios (Irrelevance Detection: Model must NOT call a tool) ─
    {
        "scenario_id": "negative_photosynthesis_explanation",
        "user_query": "Explain how photosynthesis converts sunlight into chemical energy in green plants.",
        "expected_tool": None,  # No tool needed
        "expected_args_subset": {},
        "mock_tool_response": None,
        "follow_up_keyword": "chlorophyll",
    },
    {
        "scenario_id": "negative_capital_geography",
        "user_query": "What is the capital city of Australia?",
        "expected_tool": None,  # No tool needed
        "expected_args_subset": {},
        "mock_tool_response": None,
        "follow_up_keyword": "Canberra",
    },
    {
        "scenario_id": "read_project_readme",
        "user_query": "Read the contents of the file 'README.md'.",
        "expected_tool": "read_file",
        "expected_args_subset": {"filepath": "README.md"},
        "mock_tool_response": {"content": "# Local AI Benchmark\n\nA test harness and leaderboard..."},
        "follow_up_keyword": "leaderboard",
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
    """Evaluates schema adherence, tool selection from an 8-tool catalog, and negative refusal."""

    def __init__(self):
        super().__init__(
            task_id="slm_tool_calling_multiturn",
            category="tool_calling",
            description="Tool selection from catalog, parameter typing, and negative refusal detection.",
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
        scenarios = TOOL_EVAL_SCENARIOS[:sample_limit] if sample_limit else TOOL_EVAL_SCENARIOS
        total = len(scenarios)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        breakdown = []

        system_prompt = (
            "You are an assistant equipped with tools. When a tool is needed, respond ONLY with "
            'a JSON object: {"name": "<tool_name>", "arguments": {<args>}}. '
            "If NO tool is needed or appropriate to answer the user query, respond directly in natural language without JSON."
        )

        for sc_idx, sc in enumerate(scenarios):
            if cancel_event and cancel_event.is_set():
                break
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Available tools:\n{json.dumps(TOOL_SPECS)}\n\nUser request: {sc['user_query']}"},
            ]

            # Turn 1: Model initiates tool call or responds in plain text
            gen1 = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen1.latency_ms
            total_tokens += gen1.tokens_generated

            tool_call = parse_tool_call(gen1.text)
            passed_scenario = False

            if sc["expected_tool"] is None:
                # Negative test: Model MUST NOT call a tool
                if tool_call is None and len(gen1.text.strip()) > 5:
                    passed_scenario = True
            else:
                # Positive test: Model MUST select the correct tool and arguments
                if tool_call and tool_call.get("name") == sc["expected_tool"]:
                    args = tool_call.get("arguments", {})
                    if isinstance(args, dict):
                        args_match = True
                        for k, v in sc["expected_args_subset"].items():
                            if k not in args or str(args[k]).lower() != str(v).lower():
                                args_match = False
                                break
                        if args_match:
                            passed_scenario = True

            if passed_scenario:
                passed += 1

            if on_sample:
                on_sample({
                    "task_id": self.task_id,
                    "category": self.category,
                    "sample_index": sc_idx + 1,
                    "total_samples": total,
                    "prompt": sc["user_query"],
                    "response": gen1.text,
                    "expected": f"Tool: {sc['expected_tool']}" if sc["expected_tool"] else "Direct answer (No tool)",
                    "extracted": json.dumps(tool_call) if tool_call else "No tool call (Text response)",
                    "passed": passed_scenario,
                    "latency_ms": gen1.latency_ms,
                    "tokens": gen1.tokens_generated,
                })

            breakdown.append({
                "scenario_id": sc["scenario_id"],
                "passed": passed_scenario,
                "latency_ms": gen1.latency_ms,
            })

        acc = (passed / total) if total > 0 else 0.0
        norm_score = acc * 100.0
        avg_lat = (total_latency / total) if total > 0 else 0.0
        avg_tps = (total_tokens / (total_latency / 1000.0)) if total_latency > 0 else 0.0

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            raw_score=round(acc, 4),
            raw_metric_name=self.raw_metric_name,
            normalized_score=round(norm_score, 2),
            total_samples=total,
            passed_samples=passed,
            avg_latency_ms=round(avg_lat, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={"scenarios": breakdown},
        )
