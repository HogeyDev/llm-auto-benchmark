"""
Tests for Official Benchmark Registry, Profiles, Stratified Sampling, and Statistical Confidence.
Verifies that all custom SLM benchmarks are completely absent from the registry,
sampling is deterministic and reproducible, and leaderboard eligibility is strictly enforced.
"""

import pytest
from slm_bench.backends.mock_backend import MockInferenceBackend
from slm_bench.benchmarks.registry import (
    get_task,
    list_available_tasks,
    get_profile_tasks,
    BENCHMARK_REGISTRY,
    EVALUATION_PROFILES,
)
from slm_bench.datasets.manager import DatasetManager, BENCHMARK_DATASET_META
from slm_bench.metrics.confidence import wilson_score_interval, compute_metric_confidence
from slm_bench.scoring.scoring_engine import ScoringEngine
from slm_bench.models.schema import TaskMetric


def test_custom_slm_benchmarks_completely_removed():
    """Verify that all old custom SLM tasks are purged from the registry."""
    available = list_available_tasks()
    assert "slm_reasoning_cot" not in available
    assert "slm_coding_agent" not in available
    assert "slm_instruction_ifeval" not in available
    assert "slm_tool_calling_multiturn" not in available

    # Verify attempting to get old tasks raises ValueError
    with pytest.raises(ValueError):
        get_task("slm_reasoning_cot")
    with pytest.raises(ValueError):
        get_task("slm_coding_agent")


def test_official_benchmarks_registered():
    """Verify that only official, established benchmarks are registered."""
    available = list_available_tasks()
    official_expected = [
        "mmlu_pro",
        "gpqa_diamond",
        "bbh",
        "gsm8k",
        "math500",
        "ifeval",
        "evalplus",
        "bfcl",
        "xstest",
    ]
    for b_id in official_expected:
        assert b_id in available


def test_evaluation_profiles():
    quick = get_profile_tasks("quick")
    standard = get_profile_tasks("standard")
    extended = get_profile_tasks("extended")

    assert "mmlu_pro" in quick
    assert "gsm8k" in quick
    assert "ifeval" in quick

    assert "bbh" in standard
    assert len(standard) > len(quick)

    assert "evalplus" in extended
    assert "bfcl" in extended
    assert "xstest" in extended
    assert len(extended) > len(standard)


def test_deterministic_stratified_sampling():
    dm = DatasetManager()
    dummy_data = [
        {"id": f"q_{i}", "category": "math" if i < 30 else ("physics" if i < 70 else "biology")}
        for i in range(100)
    ]

    # Two runs with same seed should yield exact same items
    sample1, ids1 = dm.sample_stratified(dummy_data, sample_size=10, seed=42, stratify_key="category")
    sample2, ids2 = dm.sample_stratified(dummy_data, sample_size=10, seed=42, stratify_key="category")

    assert ids1 == ids2
    assert len(sample1) == 10

    # Ensure categories are represented (stratification)
    cats = {item["category"] for item in sample1}
    assert "math" in cats
    assert "physics" in cats
    assert "biology" in cats

    # Different seed yields different deterministic order
    _, ids3 = dm.sample_stratified(dummy_data, sample_size=10, seed=999, stratify_key="category")
    assert ids1 != ids3


def test_wilson_confidence_intervals():
    ci_l, ci_u = wilson_score_interval(80, 100, confidence=0.95)
    assert 0.70 <= ci_l <= 0.75
    assert 0.85 <= ci_u <= 0.90
    assert ci_l < ci_u

    # Edge cases
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    assert wilson_score_interval(0, 10)[0] == 0.0
    assert wilson_score_interval(10, 10)[1] == 1.0


def test_leaderboard_eligibility_enforcement():
    engine = ScoringEngine()

    full_metric = TaskMetric(
        task_id="mmlu_pro",
        category="reasoning",
        status="FULL",
        raw_score=0.75,
        normalized_score=75.0,
        total_samples=100,
        passed_samples=75,
    )

    partial_metric = TaskMetric(
        task_id="gsm8k",
        category="mathematics",
        status="PARTIAL",
        raw_score=0.80,
        normalized_score=80.0,
        total_samples=10,
        passed_samples=8,
    )

    # 1. Full standard evaluation
    status_full, is_eligible_full = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric},
        profile="standard",
        sample_limit=None,
    )
    assert status_full == "FULL"
    assert is_eligible_full is True

    # 2. Sampled evaluation (quick profile)
    status_quick, is_eligible_quick = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric},
        profile="quick",
    )
    assert status_quick == "PARTIAL"
    assert is_eligible_quick is False

    # 3. Partial sample limit specified
    status_samp, is_eligible_samp = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric},
        profile="standard",
        sample_limit=50,
    )
    assert status_samp == "PARTIAL"
    assert is_eligible_samp is False

    # 4. Contains a partial task
    status_has_partial, is_eligible_has_partial = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric, "gsm8k": partial_metric},
        profile="standard",
        sample_limit=None,
    )
    assert status_has_partial == "PARTIAL"
    assert is_eligible_has_partial is False


def test_official_adapter_execution_with_mock():
    backend = MockInferenceBackend(model_id="mock-model", simulated_latency_ms=1.0)
    backend.load()

    # MMLU-Pro mock test with small sample
    mmlu = get_task("mmlu_pro")
    m_res = mmlu.run(backend, sample_limit=2)
    assert m_res.task_id == "mmlu_pro"
    assert m_res.status == "PARTIAL"  # Sample limit automatically marks status as PARTIAL!
    assert len(m_res.question_results) == 2

    # GPQA Diamond mock test with small sample
    gpqa = get_task("gpqa_diamond")
    g_res = gpqa.run(backend, sample_limit=2)
    assert g_res.task_id == "gpqa_diamond"
    assert g_res.status == "PARTIAL"
    assert len(g_res.question_results) == 2

    # GSM8K mock test with small sample
    gsm8k = get_task("gsm8k")
    gs_res = gsm8k.run(backend, sample_limit=2)
    assert gs_res.task_id == "gsm8k"
    assert gs_res.status == "PARTIAL"
    assert len(gs_res.question_results) == 2
