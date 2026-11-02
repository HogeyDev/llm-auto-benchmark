"""
vLLM backend for high-throughput batch evaluation with AWQ/GPTQ/FP8 quantization.
"""

from typing import List, Dict, Any, Optional
import time
from .base import BaseInferenceBackend, GenerationResult
from ..utils.logger import logger


class VLLMBackend(BaseInferenceBackend):
    """
    vLLM inference backend for high-concurrency evaluation with AWQ / GPTQ / FP8.
    """

    def __init__(
        self,
        model_id: str,
        quantization: str = "AWQ",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        super().__init__(model_id, quantization, **kwargs)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self.sampling_params = None

    def load(self) -> None:
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMBackend. "
                "Install it with 'pip install vllm'."
            )

        logger.info(f"Initializing vLLM engine for {self.model_id} (Quant: {self.quantization})...")
        vllm_quant = None
        quant_lower = self.quantization.lower()
        if "awq" in quant_lower:
            vllm_quant = "awq"
        elif "gptq" in quant_lower:
            vllm_quant = "gptq"
        elif "fp8" in quant_lower:
            vllm_quant = "fp8"

        self.llm = LLM(
            model=self.model_id,
            quantization=vllm_quant,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
        )
        self.is_loaded = True
        logger.info(f"Successfully initialized vLLM for {self.model_id}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        from vllm import SamplingParams

        if not self.is_loaded or self.llm is None:
            self.load()

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=stop_sequences or [],
        )

        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling_params)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        output = outputs[0]
        text = output.outputs[0].text
        tokens_gen = len(output.outputs[0].token_ids)

        return GenerationResult(text.strip(), dt_ms, tokens_gen)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        # Format messages into standard chat prompt
        prompt = "\n".join([f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>" for m in messages])
        prompt += "\n<|im_start|>assistant\n"
        return self.generate(prompt, max_new_tokens, temperature, stop_sequences)

    def unload(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.is_loaded = False
