"""
Backend registry and factory.
"""

from typing import Dict, Type
from .base import BaseInferenceBackend
from .mock_backend import MockInferenceBackend
from .transformers_backend import TransformersBackend
from .llamacpp_backend import LlamaCppBackend
from .vllm_backend import VLLMBackend


BACKEND_REGISTRY: Dict[str, Type[BaseInferenceBackend]] = {
    "mock": MockInferenceBackend,
    "transformers": TransformersBackend,
    "hf": TransformersBackend,
    "llamacpp": LlamaCppBackend,
    "llama_cpp": LlamaCppBackend,
    "gguf": LlamaCppBackend,
    "vllm": VLLMBackend,
}


def get_backend(
    backend_name: str,
    model_id: str,
    quantization: str = "BF16",
    **kwargs
) -> BaseInferenceBackend:
    """Instantiate the requested inference backend."""
    b_name = backend_name.lower().strip()
    if b_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Supported backends: {list(BACKEND_REGISTRY.keys())}"
        )
    backend_cls = BACKEND_REGISTRY[b_name]
    return backend_cls(model_id=model_id, quantization=quantization, **kwargs)
