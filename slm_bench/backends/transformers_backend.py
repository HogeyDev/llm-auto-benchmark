"""
Hugging Face Transformers inference backend with bitsandbytes 4-bit/8-bit and bfloat16 support.
"""

from typing import List, Dict, Any, Optional
import time
from .base import BaseInferenceBackend, GenerationResult
from ..utils.logger import logger


class TransformersBackend(BaseInferenceBackend):
    """
    Transformers backend optimized for running <=50B parameter models
    using bitsandbytes NF4, 8-bit, or native bfloat16 precision with device_map="auto".
    """

    def __init__(
        self,
        model_id: str,
        quantization: str = "BF16",
        device: str = "auto",
        trust_remote_code: bool = True,
        **kwargs
    ):
        super().__init__(model_id, quantization, device, **kwargs)
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_upper = self.quantization.upper()
        logger.info(f"Loading {self.model_id} with Transformers backend (Quant: {quant_upper}, Device: {self.device})...")

        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device,
        }

        if "4BIT" in quant_upper or "NF4" in quant_upper:
            try:
                from transformers import BitsAndBytesConfig
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            except ImportError:
                logger.warning("bitsandbytes not installed. Falling back to bfloat16/float16.")
                model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif "8BIT" in quant_upper:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes not installed. Falling back to float16.")
                model_kwargs["torch_dtype"] = torch.float16
        elif "BF16" in quant_upper and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.is_loaded = True
        logger.info(f"Successfully loaded {self.model_id}.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        import torch
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        dt_ms = (time.perf_counter() - t0) * 1000.0
        generated_tokens = output_ids[0][input_len:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return GenerationResult(output_text.strip(), dt_ms, len(generated_tokens))

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        if not self.is_loaded or self.tokenizer is None:
            self.load()

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]) + "\nassistant: "

        return self.generate(prompt, max_new_tokens, temperature, stop_sequences)

    def unload(self) -> None:
        import gc
        import torch
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded {self.model_id}.")
