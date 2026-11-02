"""
Tests for expanded benchmark adapters, registry, and inference gateway.
"""

import pytest
from slm_bench.backends.mock_backend import MockInferenceBackend
from slm_bench.execution.inference_gateway import InferenceGateway
from slm_bench.benchmarks.registry import (
    get_task,
    list_available_tasks,
    get_profile_tasks,
    BENCHMARK_REGISTRY,
    EVALUATION_PROFILES,
)
from slm_bench.metrics.confidence import wilson_score_interval, compute_metric_confidence
from slm_bench.metrics.telemetry import PerformanceCollector, TelemetrySnapshot
from slm_bench.scoring.scoring_engine import ScoringEngine
from slm_bench.scoring.quantization import compare_quantizations


def test_wilson_confidence_intervals():
    # 80/100 should be centered around 0.80 with lower ~0.71 and upper ~0.87
    ci_l, ci_u = wilson_score_interval(80, 100, confidence=0.95)
    assert 0.70 <= ci_l <= 0.75
    assert 0.85 <= ci_u <= 0.90
    assert ci_l < ci_u

    # Edge cases
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    assert wilson_score_interval(0, 10)[0] == 0.0
    assert wilson_score_interval(10, 10)[1] == 1.0

    summary = compute_metric_confidence(85, 100)
    assert summary["accuracy"] == 0.85
    assert summary["ci_range"] > 0
    assert summary["passed_samples"] == 85


def test_performance_collector():
    collector = PerformanceCollector()
    snap = TelemetrySnapshot(
        ttft_ms=120.0,
        decode_tokens_per_sec=45.0,
        tpot_ms=22.2,
        inter_token_latencies_ms=[22.0, 23.0, 21.5],
        peak_ram_mb=4096.0,
        peak_vram_mb=2048.0,
        tokens_prompt=30,
        tokens_generated=50,
        total_duration_ms=1100.0,
    )
    collector.record_snapshot(snap)
    summary = collector.get_aggregate_summary()
    assert summary["avg_ttft_ms"] == 120.0
    assert summary["avg_decode_tokens_per_sec"] == 45.0
    assert summary["total_calls"] == 1
    assert summary["total_tokens_generated"] == 50


def test_inference_gateway_telemetry():
    backend = MockInferenceBackend(model_id="test_model", simulated_latency_ms=5.0)
    gateway = InferenceGateway(backend=backend)
    
    assert gateway.capabilities["supports_chat"] is True
    res = gateway.chat([{"role": "user", "content": "What is 2+2?"}])
    assert len(res.text) > 0

    agg = gateway.collector.get_aggregate_summary()
    assert agg["total_calls"] == 1


def test_benchmark_registry_and_profiles():
    tasks = list_available_tasks()
    assert "mmlu_pro" in tasks
    assert "gpqa_diamond" in tasks
    assert "bbh" in tasks
    assert "gsm8k" in tasks
    assert "math500" in tasks
    assert "evalplus" in tasks
    assert "livecodebench" in tasks
    assert "ifbench" in tasks
    assert "bfcl" in tasks
    assert "ruler" in tasks
    assert "safety" in tasks

    quick_tasks = get_profile_tasks("quick")
    assert len(quick_tasks) >= 4
    standard_tasks = get_profile_tasks("standard")
    assert len(standard_tasks) > len(quick_tasks)
    extended_tasks = get_profile_tasks("extended")
    assert len(extended_tasks) >= len(standard_tasks)


def test_reasoning_adapters_run():
    backend = MockInferenceBackend(model_id="test_mock", simulated_latency_ms=1.0)

    # 1. MMLU-Pro
    mmlu = get_task("mmlu_pro")
    m_metric = mmlu.run(backend, sample_limit=2)
    assert m_metric.task_id == "mmlu_pro"
    assert m_metric.category == "reasoning"
    assert m_metric.total_samples == 2

    # 2. GPQA Diamond
    gpqa = get_task("gpqa_diamond")
    g_metric = gpqa.run(backend, sample_limit=2)
    assert g_metric.task_id == "gpqa_diamond"
    assert g_metric.total_samples == 2

    # 3. BBH
    bbh = get_task("bbh")
    b_metric = bbh.run(backend, sample_limit=2)
    assert b_metric.task_id == "bbh"
    assert b_metric.total_samples == 2


def test_mathematics_adapters_run():
    backend = MockInferenceBackend(model_id="test_mock", simulated_latency_ms=1.0)

    # GSM8K
    gsm = get_task("gsm8k")
    metric = gsm.run(backend, sample_limit=2)
    assert metric.task_id == "gsm8k"
    assert metric.category == "reasoning"
    assert metric.total_samples == 2

    # MATH-500
    math500 = get_task("math500")
    m_metric = math500.run(backend, sample_limit=2)
    assert m_metric.task_id == "math500"
    assert m_metric.total_samples == 2


def test_coding_and_tool_adapters_run():
    backend = MockInferenceBackend(model_id="test_mock", simulated_latency_ms=1.0)

    # EvalPlus
    evalplus = get_task("evalplus")
    ep_metric = evalplus.run(backend, sample_limit=2)
    assert ep_metric.task_id == "evalplus_coding"
    assert ep_metric.category == "coding"

    # LiveCodeBench
    lcb = get_task("livecodebench")
    lcb_metric = lcb.run(backend, sample_limit=1)
    assert lcb_metric.task_id == "livecodebench"

    # BFCL
    bfcl = get_task("bfcl")
    bfcl_metric = bfcl.run(backend, sample_limit=2)
    assert bfcl_metric.task_id == "bfcl_v4"
    assert bfcl_metric.category == "tool_calling"


def test_instruction_context_safety_adapters_run():
    backend = MockInferenceBackend(model_id="test_mock", simulated_latency_ms=1.0)

    # IFBench
    ifbench = get_task("ifbench")
    if_metric = ifbench.run(backend, sample_limit=2)
    assert if_metric.task_id == "ifbench"
    assert if_metric.category == "instruction"

    # RULER
    ruler = get_task("ruler")
    r_metric = ruler.run(backend, sample_limit=1)
    assert r_metric.task_id == "ruler_context"

    # Safety
    safety = get_task("safety")
    s_metric = safety.run(backend, sample_limit=2)
    assert s_metric.task_id == "safety_reliability"


def test_scoring_engine_and_indices():
    scoring = ScoringEngine()
    backend = MockInferenceBackend(model_id="test_mock", simulated_latency_ms=1.0)
    
    tasks = {
        "reasoning": get_task("slm_reasoning_cot").run(backend, sample_limit=2),
        "coding": get_task("slm_coding_agent").run(backend, sample_limit=2),
        "instruction": get_task("slm_instruction_ifeval").run(backend, sample_limit=2),
        "tool_calling": get_task("slm_tool_calling_multiturn").run(backend, sample_limit=2),
    }

    categories = scoring.aggregate_categories(tasks)
    assert len(categories) >= 4
    slm_idx = scoring.compute_slm_index(categories)
    assert 0.0 <= slm_idx <= 100.0

    coding_agent_idx = scoring.compute_coding_agentic_index(categories)
    assert 0.0 <= coding_agent_idx <= 100.0

    eff = scoring.compute_local_efficiency_index(
        slm_index=slm_idx,
        avg_tokens_per_sec=55.0,
        model_ram_gb=4.0,
    )
    assert eff["score_per_gb"] > 0
    assert eff["tokens_per_sec"] == 55.0
