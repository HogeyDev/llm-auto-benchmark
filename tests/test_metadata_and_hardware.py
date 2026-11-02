"""
Tests for hardware detection, model metadata, and parameter extraction.
"""

import pytest
from slm_bench.models.hardware import get_hardware_info
from slm_bench.models.metadata import (
    determine_param_tier,
    extract_params_from_name,
    generate_model_slug,
    fetch_hf_model_metadata,
)


def test_determine_param_tier():
    assert determine_param_tier(1.5) == "<3B"
    assert determine_param_tier(3.0) == "3B-8B"
    assert determine_param_tier(7.0) == "3B-8B"
    assert determine_param_tier(14.0) == "8B-14B"
    assert determine_param_tier(32.0) == "14B-35B"
    assert determine_param_tier(45.0) == "35B-50B"
    assert determine_param_tier(70.0) == ">50B"
    assert determine_param_tier(None) == "Unknown"


def test_extract_params_from_name():
    assert extract_params_from_name("Qwen2.5-7B-Instruct") == 7.0
    assert extract_params_from_name("Phi-3.5-mini-instruct-3.8b") == 3.8
    assert extract_params_from_name("Llama-3.2-1B") == 1.0
    assert extract_params_from_name("Qwen2.5-0.5B-Chat") == 0.5
    assert extract_params_from_name("SmolLM-360M") == 0.36
    assert extract_params_from_name("NoParamName") is None


def test_generate_model_slug():
    slug = generate_model_slug("meta-llama/Llama-3.2-3B-Instruct", "BF16")
    assert slug == "Llama-3.2-3B-Instruct-BF16"
    assert "/" not in slug
    assert " " not in slug

    slug2 = generate_model_slug("Qwen/Qwen2.5-Coder-7B-Instruct", "GGUF_Q4_K_M")
    assert slug2 == "Qwen2.5-Coder-7B-Instruct-GGUF_Q4_K_M"


def test_get_hardware_info():
    hw = get_hardware_info()
    assert hw.os is not None
    assert hw.cpu is not None
    assert hw.cpu_cores > 0
    assert hw.total_ram_gb > 0


def test_fetch_hf_model_metadata_fallback():
    # Test fallback heuristics when offline or with custom name
    meta = fetch_hf_model_metadata("local/test-model-7B", quantization="AWQ_4BIT", backend="vllm")
    assert meta.model_id == "local/test-model-7B"
    assert meta.param_count_b == 7.0
    assert meta.param_tier == "3B-8B"
    assert meta.quantization == "AWQ_4BIT"
    assert meta.backend == "vllm"
    assert meta.model_ram_gb is not None


def test_estimate_model_ram_gb():
    from slm_bench.models.metadata import estimate_model_ram_gb
    assert estimate_model_ram_gb(7.0, "GGUF_Q4_K_M") == 4.55
    assert estimate_model_ram_gb(8.0, "GGUF_Q8_0") == 9.20
    assert estimate_model_ram_gb(3.0, "BF16") == 6.45
    assert estimate_model_ram_gb(None, "BF16") is None


def test_auto_detect_model_ram_gb(tmp_path):
    from slm_bench.models.memory_detector import auto_detect_model_ram_gb, detect_local_file_ram_gb

    # Test file-based detection on dummy GGUF file
    dummy_gguf = tmp_path / "model.gguf"
    # Write 10 MB file
    dummy_gguf.write_bytes(b"\x00" * (10 * 1024 * 1024))
    file_ram = detect_local_file_ram_gb(str(dummy_gguf))
    assert file_ram is not None
    assert round(file_ram, 2) == 0.01

    # Test fallback analytical detection
    ram = auto_detect_model_ram_gb(param_count_b=7.0, quantization="GGUF_Q4_K_M")
    assert ram == 4.55
