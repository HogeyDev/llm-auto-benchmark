"""
Tests for inference backend abstraction and registry.
"""

import pytest
from slm_bench.backends import get_backend, MockInferenceBackend, BACKEND_REGISTRY


def test_backend_registry():
    assert "mock" in BACKEND_REGISTRY
    assert "transformers" in BACKEND_REGISTRY
    assert "llamacpp" in BACKEND_REGISTRY
    assert "vllm" in BACKEND_REGISTRY


def test_mock_backend_generation():
    backend = get_backend("mock", model_id="test/mock-model", quantization="BF16")
    assert isinstance(backend, MockInferenceBackend)
    backend.load()
    assert backend.is_loaded

    # Math test prompt
    res = backend.generate("Calculate 12 cakes per hour baker")
    assert res.text != ""
    assert "#### 13" in res.text
    assert res.latency_ms >= 0

    # Code prompt
    res_code = backend.generate("Write a python function fibonacci")
    assert "def fibonacci" in res_code.text

    # Tool calling
    res_tool = backend.generate("What is the weather in Tokyo?")
    assert "get_weather" in res_tool.text

    backend.unload()
    assert not backend.is_loaded


def test_invalid_backend():
    with pytest.raises(ValueError):
        get_backend("nonexistent_backend", model_id="test")
