"""
llama.cpp / GGUF backend for memory-efficient consumer GPU and CPU offloading.
"""

from typing import List, Dict, Any, Optional
import time
from .base import BaseInferenceBackend, GenerationResult
from ..utils.logger import logger


class LlamaCppBackend(BaseInferenceBackend):
    """
    llama.cpp backend supporting GGUF quantized models (e.g. Q4_K_M, Q5_K_M, Q8_0).
    Provides fine-grained GPU layer offloading (n_gpu_layers) to fit up to 32B models
    in constrained VRAM setups.
    """

    def __init__(
        self,
        model_id: str,
        quantization: str = "GGUF_Q4_K_M",
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, quantization, **kwargs)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.model_path = model_path or model_id
        self.llm = None

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for the llamacpp backend. "
                "Install it with 'pip install llama-cpp-python'."
            )

        logger.info(f"Initializing LlamaCppBackend from {self.model_path} (n_gpu_layers={self.n_gpu_layers})...")
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        self.is_loaded = True
        logger.info(f"Loaded GGUF model: {self.model_path}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        if not self.is_loaded or self.llm is None:
            self.load()

        t0 = time.perf_counter()
        resp = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_sequences or [],
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        text = resp["choices"][0]["text"]
        usage = resp.get("usage", {})
        tokens_gen = usage.get("completion_tokens", len(text.split()))

        return GenerationResult(text.strip(), dt_ms, tokens_gen)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        if not self.is_loaded or self.llm is None:
            self.load()

        t0 = time.perf_counter()
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_sequences or [],
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp["choices"][0]
        text = choice.get("message", {}).get("content", "")
        usage = resp.get("usage", {})
        tokens_gen = usage.get("completion_tokens", len(text.split()))

        return GenerationResult(text.strip(), dt_ms, tokens_gen)

    def unload(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.is_loaded = False
