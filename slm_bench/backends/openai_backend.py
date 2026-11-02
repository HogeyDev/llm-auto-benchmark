"""
OpenAI-compatible inference backend for SLM-Bench.
Supports local engines like Ollama, LM Studio, vLLM, LocalAI, Jan, text-generation-webui,
or any custom runner exposing an OpenAI-compatible REST API (/v1/chat/completions or /v1/completions).
"""

import json
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from .base import BaseInferenceBackend, GenerationResult
from ..utils.logger import logger


class OpenAICompatibleBackend(BaseInferenceBackend):
    """
    Inference backend communicating with an OpenAI-compatible REST API.
    Ideal for testing local models hosted via Ollama, LM Studio, vLLM,
    or custom software runtime environments.
    """

    def __init__(
        self,
        model_id: str,
        quantization: str = "Custom / Server",
        device: str = "auto",
        endpoint_url: str = "http://localhost:11434/v1",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_id=model_id, quantization=quantization, device=device, **kwargs)
        # Normalize endpoint URL (strip trailing slashes)
        base = endpoint_url or kwargs.get("base_url") or "http://localhost:11434/v1"
        self.endpoint_url = base.rstrip("/")
        self.api_key = api_key or kwargs.get("token") or "EMPTY"
        self.timeout = float(timeout)
        self.reasoning_effort = reasoning_effort or kwargs.get("reasoning_effort")
        self.model_ram_gb: Optional[float] = kwargs.get("model_ram_gb")
        self.server_engine: Optional[str] = None
        self.server_version: Optional[str] = None
        self.server_runtime: Optional[str] = None

    def load(self) -> None:
        """Verify endpoint connectivity, model availability, and runtime environment if supported."""
        logger.info(f"Connecting to OpenAI-compatible endpoint at: [cyan]{self.endpoint_url}[/cyan] for model '[bold green]{self.model_id}[/bold green]'...")
        # Optional ping to /models
        try:
            req = urllib.request.Request(
                f"{self.endpoint_url}/models",
                headers=self._get_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
                    for m in data.get("data", []):
                        if isinstance(m, dict) and m.get("owned_by"):
                            self.server_engine = m.get("owned_by")
                            break
                    logger.info(f"Connected to endpoint successfully. Found {len(models)} model(s).")
        except Exception as e:
            logger.debug(f"Endpoint /models check note (endpoint might only serve completions directly): {e}")

        # Optional probe to /version (e.g. vLLM /version endpoint)
        try:
            base_url = self.endpoint_url.rsplit("/v1", 1)[0]
            req_ver = urllib.request.Request(
                f"{base_url}/version",
                headers=self._get_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req_ver, timeout=2) as resp:
                if resp.status == 200:
                    v_data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(v_data, dict) and "version" in v_data:
                        self.server_version = v_data["version"]
        except Exception:
            pass

        # Synthesize server runtime descriptor
        from ..models.hardware import is_endpoint_wsl_relayed, is_wsl_environment
        engine_str = self.server_engine or "OpenAI API"
        if engine_str.lower() == "vllm":
            engine_str = "vLLM"
        if self.server_version:
            engine_str = f"{engine_str} {self.server_version}"
        if is_endpoint_wsl_relayed(self.endpoint_url) or is_wsl_environment():
            engine_str = f"{engine_str} in WSL2"
        self.server_runtime = engine_str

        self.is_loaded = True
        logger.info(f"OpenAI-compatible backend loaded ({self.server_runtime}) and ready for '{self.model_id}'.")

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "slm-bench/0.1.0",
        }
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON POST request to the OpenAI-compatible endpoint."""
        url = f"{self.endpoint_url}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._get_headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"HTTP error {e.code} from {url}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI-compatible endpoint at {url}: {e}") from e

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text completion from a prompt using the chat endpoint."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate completion using /chat/completions with fallback to /completions."""
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": max(0.0, temperature),
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if stop_sequences:
            payload["stop"] = stop_sequences

        # 1. Real-time token streaming if callback is registered
        if getattr(self, "on_stream_chunk", None):
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            url = f"{self.endpoint_url}/chat/completions"
            data = json.dumps(stream_payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=self._get_headers(), method="POST")
            start_time = time.perf_counter()
            collected_chunks: List[str] = []
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for line in resp:
                        line_str = line.decode("utf-8").strip()
                        if not line_str or line_str.startswith(":"):
                            continue
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            chunk_json = line_str[6:]
                            try:
                                chunk_data = json.loads(chunk_json)
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content") or ""
                                    if content:
                                        collected_chunks.append(content)
                                        self.on_stream_chunk(content)
                            except Exception:
                                pass
                full_text = "".join(collected_chunks)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                tokens = max(1, len(full_text.split()))
                res = GenerationResult(text=full_text, latency_ms=latency_ms, tokens_generated=tokens)
                if getattr(self, "on_generation_result", None):
                    self.on_generation_result({"text": full_text, "latency_ms": latency_ms, "tokens": tokens})
                return res
            except Exception as e:
                logger.debug(f"Streaming probe note, falling back to synchronous execution: {e}")

        start_time = time.perf_counter()
        try:
            resp_data = self._post("chat/completions", payload)
        except Exception as e:
            # Fallback attempt to /completions for non-chat server runners
            try:
                prompt_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
                payload_comp = {
                    "model": self.model_id,
                    "prompt": prompt_text,
                    "max_tokens": max_new_tokens,
                    "temperature": max(0.0, temperature),
                }
                if stop_sequences:
                    payload_comp["stop"] = stop_sequences
                resp_data = self._post("completions", payload_comp)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                choices = resp_data.get("choices", [])
                text = choices[0].get("text", "") if choices else ""
                tokens = resp_data.get("usage", {}).get("completion_tokens", max(1, len(text.split())))
                return GenerationResult(text=text, latency_ms=latency_ms, tokens_generated=tokens)
            except Exception:
                raise e

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        choices = resp_data.get("choices", [])
        if not choices:
            return GenerationResult(text="", latency_ms=latency_ms, tokens_generated=0)

        first_choice = choices[0]
        message = first_choice.get("message", {})
        text = message.get("content") or ""

        usage = resp_data.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None or completion_tokens == 0:
            completion_tokens = max(1, len(text.split()))

        if getattr(self, "on_stream_chunk", None):
            self.on_stream_chunk(text)
        if getattr(self, "on_generation_result", None):
            self.on_generation_result({"text": text, "latency_ms": latency_ms, "tokens": completion_tokens})

        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            tokens_generated=completion_tokens,
        )

    def get_model_ram_gb(self) -> Optional[float]:
        """Automatically retrieve model RAM/VRAM footprint from Ollama or local server."""
        if self.model_ram_gb:
            return self.model_ram_gb
        from ..models.memory_detector import detect_ollama_model_ram_gb, detect_runner_process_ram_gb
        ram = detect_ollama_model_ram_gb(self.endpoint_url, self.model_id)
        if ram:
            return ram
        return detect_runner_process_ram_gb()

    def unload(self) -> None:
        """Release backend resources."""
        self.is_loaded = False
