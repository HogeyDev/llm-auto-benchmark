"""
Tests for benchmark tasks and execution logic.
"""

from slm_bench.backends.mock_backend import MockInferenceBackend
from slm_bench.benchmarks.reasoning_math import (
    ReasoningBenchmark,
    extract_numeric_answer,
    extract_reasoning_answer,
    REASONING_DATASET,
)
from slm_bench.benchmarks.coding_agent import (
    CodingBenchmark,
    extract_python_code,
    run_code_in_sandbox,
    CODING_DATASET,
)
from slm_bench.benchmarks.instruction_following import (
    InstructionFollowingBenchmark,
    verify_json_schema,
    verify_markdown_table,
    verify_bullet_count,
    verify_word_count,
    verify_negative_constraint,
    INSTRUCTION_DATASET,
)
from slm_bench.benchmarks.tool_calling import (
    ToolCallingBenchmark,
    parse_tool_call,
    TOOL_EVAL_SCENARIOS,
    TOOL_SPECS,
)


def test_extract_numeric_answer():
    assert extract_numeric_answer("We calculate the total. #### 42") == "42"
    assert extract_numeric_answer("#### -12.5") == "-12.5"
    assert extract_numeric_answer("The final answer is 100") == "100"
    assert extract_numeric_answer("The solution is \\boxed{768}.") == "768"


def test_extract_reasoning_answer_arc_multiple_choice():
    # Test ARC extraction
    assert extract_reasoning_answer("The copper penny tests Mohs hardness. #### C", item_type="multiple_choice") == "C"
    assert extract_reasoning_answer("Therefore \\boxed{B}", item_type="multiple_choice") == "B"
    assert extract_reasoning_answer("The correct option is (A)", item_type="multiple_choice") == "A"
    assert extract_reasoning_answer("After reviewing the laws, the answer is D.", item_type="multiple_choice") == "D"


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


def test_ifeval_validators():
    assert verify_markdown_table("| Col1 | Col2 | Col3 |\n|---|---|---|\n| a | b | c |\n| d | e | f |")
    assert not verify_markdown_table("Not a table")
    assert verify_bullet_count("- item 1\n- item 2\n- item 3", 3)
    assert not verify_bullet_count("- item 1\n- item 2", 3)
    assert verify_word_count("one two three four five", 4, 6)
    assert not verify_word_count("one two", 4, 6)
    assert verify_negative_constraint("a quick brown fox jumps", "e")
    assert not verify_negative_constraint("the brown fox", "e")
    assert verify_json_schema('{"host": "api.com", "port": 80, "ssl": true, "tags": ["a"]}', {"host": str, "port": int, "ssl": bool, "tags": list})


def test_benchmark_dataset_sizes():
    assert len(REASONING_DATASET) >= 20
    assert len(CODING_DATASET) >= 15
    assert len(INSTRUCTION_DATASET) >= 15
    assert len(TOOL_EVAL_SCENARIOS) >= 10
    assert len(TOOL_SPECS) >= 6


def test_all_benchmarks_with_mock():
    backend = MockInferenceBackend("test/model")
    backend.load()

    reasoning_task = ReasoningBenchmark()
    res_r = reasoning_task.run(backend, sample_limit=5)
    assert 0.0 <= res_r.normalized_score <= 100.0
    assert res_r.total_samples == 5

    coding_task = CodingBenchmark()
    res_c = coding_task.run(backend, sample_limit=5)
    assert 0.0 <= res_c.normalized_score <= 100.0
    assert res_c.total_samples == 5

    instruction_task = InstructionFollowingBenchmark()
    res_i = instruction_task.run(backend, sample_limit=5)
    assert 0.0 <= res_i.normalized_score <= 100.0
    assert res_i.total_samples == 5

    tool_task = ToolCallingBenchmark()
    res_t = tool_task.run(backend, sample_limit=5)
    assert 0.0 <= res_t.normalized_score <= 100.0
    assert res_t.total_samples == 5
