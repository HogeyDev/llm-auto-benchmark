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
    assert "openai" in BACKEND_REGISTRY
    assert "ollama" in BACKEND_REGISTRY
    assert "lmstudio" in BACKEND_REGISTRY
    assert "endpoint" in BACKEND_REGISTRY


def test_openai_compatible_backend_initialization_and_mock(monkeypatch):
    from slm_bench.backends.openai_backend import OpenAICompatibleBackend
    backend = get_backend(
        "openai",
        model_id="qwen2.5:7b",
        endpoint_url="http://localhost:11434/v1",
        api_key="test-key",
        model_ram_gb=4.5,
    )
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.endpoint_url == "http://localhost:11434/v1"
    assert backend.api_key == "test-key"
    assert backend.model_ram_gb == 4.5

    # Mock _post to simulate OpenAI chat completions API response
    def mock_post(path, payload):
        return {
            "id": "chatcmpl-123",
            "choices": [{
                "message": {"role": "assistant", "content": "The answer is #### 42"}
            }],
            "usage": {"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15}
        }

    monkeypatch.setattr(backend, "_post", mock_post)
    backend.load()
    assert backend.is_loaded

    gen_res = backend.generate("Solve 20 + 22")
    assert "#### 42" in gen_res.text
    assert gen_res.tokens_generated == 10
    assert gen_res.latency_ms >= 0


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
