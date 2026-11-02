"""
Function calling and multi-turn tool interaction benchmark task.
Evaluates:
- Tool selection from an 8-tool catalog
- Strict parameter schema & type validation
- Irrelevance / negative testing (knowing when NOT to invoke a tool)
- Multi-turn execution: feeding returned tool output back to model
- Follow-up synthesis: checking if model correctly uses tool results
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
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


def validate_schema_compliance(tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that arguments adhere strictly to the JSON schema defined in TOOL_SPECS.
    Checks required parameters, property types, and enum restrictions.
    """
    spec = next((t["function"] for t in TOOL_SPECS if t["function"]["name"] == tool_name), None)
    if not spec:
        return False, f"Unknown tool: '{tool_name}'"

    params = spec.get("parameters", {})
    required = params.get("required", [])
    properties = params.get("properties", {})

    # Check required fields
    for req in required:
        if req not in arguments:
            return False, f"Missing required parameter '{req}'"

    # Validate types and enum constraints
    for k, v in arguments.items():
        if k not in properties:
            continue
        expected_type = properties[k].get("type")
        if expected_type == "string" and not isinstance(v, str):
            return False, f"Parameter '{k}' expected string, got {type(v).__name__}"
        elif expected_type == "number" and not isinstance(v, (int, float)):
            return False, f"Parameter '{k}' expected number, got {type(v).__name__}"
        elif expected_type == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
            return False, f"Parameter '{k}' expected integer, got {type(v).__name__}"

        enum_vals = properties[k].get("enum")
        if enum_vals and v not in enum_vals:
            return False, f"Parameter '{k}' value '{v}' not in allowed enums {enum_vals}"

    return True, None


def get_stratified_scenarios(scenarios: List[Dict[str, Any]], sample_limit: Optional[int]) -> List[Dict[str, Any]]:
    """
    Ensure balanced representation of positive and negative tests when sample_limit is used,
    preventing negative tests from being completely truncated.
    """
    if not sample_limit or sample_limit >= len(scenarios):
        return scenarios

    positives = [s for s in scenarios if s["expected_tool"] is not None]
    negatives = [s for s in scenarios if s["expected_tool"] is None]

    if not negatives:
        return scenarios[:sample_limit]

    # Reserve at least 1 negative if sample_limit >= 2
    neg_count = max(1, int(round(sample_limit * (len(negatives) / len(scenarios)))))
    pos_count = max(1, sample_limit - neg_count)

    return positives[:pos_count] + negatives[:neg_count]


class ToolCallingBenchmark(BaseBenchmarkTask):
    """
    Evaluates:
    1. Tool Selection: Identifies correct function from catalog.
    2. Schema Compliance: Adheres to argument types and enum constraints.
    3. Argument Extraction: Matches required domain parameters.
    4. Negative Refusal: Refrains from calling tools when direct answer is appropriate.
    5. Multi-turn Synthesis: Incorporates returned tool execution results into final answer.
    """

    def __init__(self):
        super().__init__(
            task_id="slm_tool_calling_multiturn",
            category="tool_calling",
            description="Tool selection from catalog, parameter typing, negative refusal, and multi-turn execution.",
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
        scenarios = get_stratified_scenarios(TOOL_EVAL_SCENARIOS, sample_limit)
        total = len(scenarios)
        passed = 0
        total_latency = 0.0
        total_tokens = 0

        # Subtest tracking counters
        selection_correct = 0
        schema_compliant = 0
        args_correct = 0
        negative_correct = 0
        multiturn_correct = 0

        positive_count = sum(1 for s in scenarios if s["expected_tool"] is not None)
        negative_count = total - positive_count

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
            turn2_text = ""

            sub_status = {
                "selection": False,
                "schema_compliance": False,
                "arguments_correct": False,
                "multiturn_synthesis": False,
                "is_negative": sc["expected_tool"] is None,
            }

            if sc["expected_tool"] is None:
                # Negative test: Model MUST NOT call a tool and must respond with relevant text
                if tool_call is None and len(gen1.text.strip()) > 5:
                    sub_status["selection"] = True
                    sub_status["arguments_correct"] = True
                    sub_status["schema_compliance"] = True
                    negative_correct += 1
                    # Check if model addressed query
                    if sc.get("follow_up_keyword", "").lower() in gen1.text.lower():
                        sub_status["multiturn_synthesis"] = True
                    passed_scenario = True
            else:
                # Positive test: Model MUST select the correct tool and arguments
                if tool_call and tool_call.get("name") == sc["expected_tool"]:
                    sub_status["selection"] = True
                    selection_correct += 1

                    args = tool_call.get("arguments", {})
                    if isinstance(args, dict):
                        # 1. Full schema compliance check
                        is_compliant, _ = validate_schema_compliance(sc["expected_tool"], args)
                        if is_compliant:
                            sub_status["schema_compliance"] = True
                            schema_compliant += 1

                        # 2. Argument value accuracy
                        args_match = True
                        for k, v in sc["expected_args_subset"].items():
                            if k not in args or str(args[k]).lower() != str(v).lower():
                                args_match = False
                                break
                        if args_match:
                            sub_status["arguments_correct"] = True
                            args_correct += 1

                    # Turn 2: Feed back mock tool response and evaluate follow-up synthesis
                    if sub_status["selection"] and sub_status["arguments_correct"] and sc.get("mock_tool_response"):
                        turn2_messages = list(messages)
                        turn2_messages.append({"role": "assistant", "content": gen1.text})
                        turn2_messages.append({
                            "role": "tool",
                            "content": f"Tool result: {json.dumps(sc['mock_tool_response'])}",
                        })

                        gen2 = backend.chat(turn2_messages, max_new_tokens=256, temperature=0.0)
                        total_latency += gen2.latency_ms
                        total_tokens += gen2.tokens_generated
                        turn2_text = gen2.text

                        # Evaluate if model synthesized tool result (checks follow-up keyword)
                        kw = sc.get("follow_up_keyword", "").lower()
                        if kw in gen2.text.lower() or kw in gen1.text.lower():
                            sub_status["multiturn_synthesis"] = True
                            multiturn_correct += 1

                    # A positive scenario passes if selection, arguments, and follow-up succeed
                    if sub_status["selection"] and sub_status["arguments_correct"]:
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
                    "follow_up_response": turn2_text,
                    "expected": f"Tool: {sc['expected_tool']}" if sc["expected_tool"] else "Direct answer (No tool)",
                    "extracted": json.dumps(tool_call) if tool_call else "No tool call (Text response)",
                    "passed": passed_scenario,
                    "sub_status": sub_status,
                    "latency_ms": gen1.latency_ms,
                    "tokens": gen1.tokens_generated,
                })

            breakdown.append({
                "scenario_id": sc["scenario_id"],
                "passed": passed_scenario,
                "sub_status": sub_status,
                "latency_ms": gen1.latency_ms,
            })

        acc = (passed / total) if total > 0 else 0.0
        norm_score = acc * 100.0
        avg_lat = (total_latency / total) if total > 0 else 0.0
        avg_tps = (total_tokens / (total_latency / 1000.0)) if total_latency > 0 else 0.0

        sub_metrics = {
            "scenarios": breakdown,
            "tool_selection_accuracy": round(selection_correct / max(1, positive_count), 4),
            "schema_compliance_rate": round(schema_compliant / max(1, positive_count), 4),
            "argument_accuracy": round(args_correct / max(1, positive_count), 4),
            "negative_refusal_accuracy": round(negative_correct / max(1, negative_count), 4) if negative_count > 0 else 1.0,
            "multiturn_synthesis_rate": round(multiturn_correct / max(1, positive_count), 4),
        }

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
            sub_metrics=sub_metrics,
        )
