"""
Tests for benchmark tasks and execution logic.
"""

from slm_bench.backends.mock_backend import MockInferenceBackend
from slm_bench.benchmarks.reasoning_math import ReasoningBenchmark, extract_numeric_answer
from slm_bench.benchmarks.coding_agent import CodingBenchmark, extract_python_code, run_code_in_sandbox
from slm_bench.benchmarks.instruction_following import InstructionFollowingBenchmark
from slm_bench.benchmarks.tool_calling import ToolCallingBenchmark, parse_tool_call


def test_extract_numeric_answer():
    assert extract_numeric_answer("We calculate the total. #### 42") == "42"
    assert extract_numeric_answer("#### -12.5") == "-12.5"
    assert extract_numeric_answer("The final answer is 100") == "100"


def test_extract_python_code():
    snippet = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    extracted = extract_python_code(snippet)
    assert extracted == "def add(a, b):\n    return a + b"


def test_run_code_in_sandbox():
    good_code = "def add(a, b):\n    return a + b"
    passed, msg = run_code_in_sandbox(good_code, "add", [("add(2, 3)", 5), ("add(-1, 1)", 0)])
    assert passed
    assert "passed" in msg

    bad_code = "def add(a, b):\n    return a - b"
    failed, msg = run_code_in_sandbox(bad_code, "add", [("add(2, 3)", 5)])
    assert not failed


def test_parse_tool_call():
    raw = 'Sure! {"name": "get_weather", "arguments": {"location": "Paris"}}'
    parsed = parse_tool_call(raw)
    assert parsed is not None
    assert parsed["name"] == "get_weather"
    assert parsed["arguments"]["location"] == "Paris"


def test_all_benchmarks_with_mock():
    backend = MockInferenceBackend("test/model")
    backend.load()

    reasoning_task = ReasoningBenchmark()
    res_r = reasoning_task.run(backend, sample_limit=2)
    assert 0.0 <= res_r.normalized_score <= 100.0
    assert res_r.total_samples == 2

    coding_task = CodingBenchmark()
    res_c = coding_task.run(backend, sample_limit=2)
    assert 0.0 <= res_c.normalized_score <= 100.0
    assert res_c.total_samples == 2

    instruction_task = InstructionFollowingBenchmark()
    res_i = instruction_task.run(backend, sample_limit=2)
    assert 0.0 <= res_i.normalized_score <= 100.0
    assert res_i.total_samples == 2

    tool_task = ToolCallingBenchmark()
    res_t = tool_task.run(backend, sample_limit=2)
    assert 0.0 <= res_t.normalized_score <= 100.0
    assert res_t.total_samples == 2
