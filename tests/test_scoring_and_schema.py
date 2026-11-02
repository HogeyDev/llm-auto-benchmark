"""
Tests for BenchmarkResult schema validation, normalization, and SLM-Index calculation.
"""

from slm_bench.models.schema import BenchmarkResult, ModelMetadata, HardwareInfo, TaskMetric, CategoryScore
from slm_bench.benchmarks.suite import BenchmarkSuite
from slm_bench.backends.mock_backend import MockInferenceBackend


def test_benchmark_result_schema_serialization():
    meta = ModelMetadata(
        model_id="test/model-3B",
        model_slug="model-3B-BF16",
        param_count_b=3.1,
        param_tier="3B-8B",
        architecture="LlamaForCausalLM",
        quantization="BF16",
        backend="mock",
    )
    hw = HardwareInfo(
        os="Windows 11",
        cpu="Intel i7",
        cpu_cores=8,
        total_ram_gb=16.0,
    )
    task_m = TaskMetric(
        task_id="slm_reasoning_cot",
        category="reasoning",
        raw_score=0.8,
        raw_metric_name="accuracy",
        normalized_score=80.0,
        total_samples=5,
        passed_samples=4,
    )
    cat_s = CategoryScore(
        category="reasoning",
        weight=0.25,
        normalized_score=80.0,
        task_count=1,
    )

    result = BenchmarkResult(
        schema_version="1.0.0",
        evaluation_id="eval-1234",
        model=meta,
        hardware=hw,
        tasks={"slm_reasoning_cot": task_m},
        categories={"reasoning": cat_s},
        slm_index=80.0,
    )

    # Test JSON serialization & roundtrip
    json_str = result.model_dump_json()
    reloaded = BenchmarkResult.model_validate_json(json_str)
    assert reloaded.evaluation_id == "eval-1234"
    assert reloaded.slm_index == 80.0
    assert reloaded.model.model_slug == "model-3B-BF16"

    # Test summary dict
    summary = result.to_summary_dict()
    assert summary["slm_index"] == 80.0
    assert summary["params_b"] == 3.1


def test_suite_composite_index_calculation():
    suite = BenchmarkSuite()
    backend = MockInferenceBackend("test/model")
    backend.load()

    meta = ModelMetadata(
        model_id="test/model-7B",
        model_slug="model-7B-BF16",
        param_count_b=7.0,
        param_tier="3B-8B",
        backend="mock",
    )
    hw = HardwareInfo(os="Linux", cpu="Xeon", cpu_cores=16, total_ram_gb=64.0)

    result = suite.run_suite(backend, meta, hw, sample_limit=2)
    assert 0.0 <= result.slm_index <= 100.0
    assert "reasoning" in result.categories
    assert "coding" in result.categories
    assert "instruction" in result.categories
    assert "tool_calling" in result.categories
