"""
Tests for BenchmarkResult Schema 2.0.0 validation, normalization, and SLM-Index calculation.
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
        model_ram_gb=3.8,
    )
    task_m = TaskMetric(
        task_id="mmlu_pro",
        category="reasoning",
        status="FULL",
        raw_score=0.8,
        raw_metric_name="accuracy",
        normalized_score=80.0,
        total_samples=100,
        passed_samples=80,
    )
    cat_s = CategoryScore(
        category="reasoning",
        weight=0.35,
        normalized_score=80.0,
        task_count=1,
        tasks=["mmlu_pro"],
    )

    result = BenchmarkResult(
        schema_version="2.0.0",
        evaluation_id="eval-1234",
        status="FULL",
        eligible_for_leaderboard=True,
        model=meta,
        hardware=hw,
        tasks={"mmlu_pro": task_m},
        categories={"reasoning": cat_s},
        slm_index=80.0,
        notes="Evaluated via local endpoint",
    )

    # Test JSON serialization & roundtrip
    json_str = result.model_dump_json()
    reloaded = BenchmarkResult.model_validate_json(json_str)
    assert reloaded.evaluation_id == "eval-1234"
    assert reloaded.schema_version == "2.0.0"
    assert reloaded.slm_index == 80.0
    assert reloaded.status == "FULL"
    assert reloaded.eligible_for_leaderboard is True
    assert reloaded.model.model_slug == "model-3B-BF16"
    assert reloaded.notes == "Evaluated via local endpoint"
    assert reloaded.hardware.model_ram_gb == 3.8

    # Test summary dict
    summary = result.to_summary_dict()
    assert summary["slm_index"] == 80.0
    assert summary["status"] == "FULL"
    assert summary["params_b"] == 3.1
    assert summary["model_ram_gb"] == 3.8
    assert summary["notes"] == "Evaluated via local endpoint"


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

    # Run quick sample
    result = suite.run_suite(backend, meta, hw, sample_limit=2)
    assert 0.0 <= result.slm_index <= 100.0
    assert "reasoning" in result.categories
    assert "mathematics" in result.categories
    assert "instruction" in result.categories
    assert result.status == "PARTIAL"
    assert result.eligible_for_leaderboard is False
