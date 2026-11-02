"""
Automatic detection and measurement of model RAM/VRAM footprint across environments.
Supports:
1. Direct backend runtime measurements (PyTorch model.get_memory_footprint, torch.cuda, GGUF mmap)
2. Local server telemetry (Ollama /api/ps and /api/show endpoints)
3. Local runner process inspection (Ollama, LM Studio, vLLM, llama-server via psutil)
4. Local model file / weight directory size calculation
5. Theoretical parameter & quantization analytical estimation fallback
"""

import os
import re
import json
import psutil
import urllib.request
from pathlib import Path
from typing import Optional, Any
from ..utils.logger import logger
from .metadata import estimate_model_ram_gb


def detect_runner_process_ram_gb(keywords=("ollama", "lmstudio", "vllm", "llama", "localai")) -> Optional[float]:
    """Inspect local running inference server processes to detect resident RAM usage."""
    try:
        max_rss = 0
        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if any(k in name for k in keywords):
                    mem = proc.info.get("memory_info")
                    if mem and mem.rss > max_rss:
                        max_rss = mem.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if max_rss > 500 * 1024 * 1024:  # At least 500 MB to represent an active model
            return round(max_rss / (1024 ** 3), 2)
    except Exception as e:
        logger.debug(f"Process RAM inspection note: {e}")
    return None


def detect_ollama_model_ram_gb(endpoint_url: str, model_id: str) -> Optional[float]:
    """Query Ollama server API (/api/ps and /api/show) for exact model VRAM/RAM allocation."""
    try:
        # Strip /v1 from the endpoint to reach Ollama native API
        base_url = re.sub(r"/v1/?$", "", endpoint_url).rstrip("/")

        # 1. Try /api/ps (currently running models in memory/VRAM)
        req = urllib.request.Request(f"{base_url}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("models", []):
                    name = (m.get("name") or "").lower()
                    m_model = (m.get("model") or "").lower()
                    target = model_id.lower()
                    if target in name or target in m_model or name in target:
                        size_vram = m.get("size_vram") or m.get("size")
                        if size_vram and size_vram > 0:
                            return round(size_vram / (1024 ** 3), 2)

        # 2. Try /api/show (model weights specification)
        payload = json.dumps({"name": model_id}).encode("utf-8")
        req_show = urllib.request.Request(
            f"{base_url}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_show, timeout=3) as resp_show:
            if resp_show.status == 200:
                data_show = json.loads(resp_show.read().decode("utf-8"))
                details = data_show.get("details", {})
                size = data_show.get("size") or details.get("size")
                if size and size > 0:
                    return round(size / (1024 ** 3), 2)
    except Exception as e:
        logger.debug(f"Ollama RAM query note: {e}")
    return None


def detect_local_file_ram_gb(model_path: str) -> Optional[float]:
    """Calculate RAM required to map local GGUF, SafeTensors, or PyTorch weights."""
    p = Path(model_path)
    if p.is_file():
        return round(p.stat().st_size / (1024 ** 3), 2)
    elif p.is_dir():
        total_bytes = sum(
            f.stat().st_size
            for f in p.glob("**/*")
            if f.is_file() and f.suffix in (".safetensors", ".bin", ".gguf", ".pt")
        )
        if total_bytes > 0:
            return round(total_bytes / (1024 ** 3), 2)
    return None


def auto_detect_model_ram_gb(
    backend_instance: Optional[Any] = None,
    backend_name: Optional[str] = None,
    model_id: Optional[str] = None,
    quantization: str = "BF16",
    param_count_b: Optional[float] = None,
    endpoint_url: Optional[str] = None,
) -> Optional[float]:
    """
    Automatically detects the model size in RAM using multiple detection mechanisms:
    1. Direct backend inquiry (if model is loaded in memory/VRAM)
    2. Ollama / custom runner REST telemetry (/api/ps, /api/show)
    3. Host runner process resident memory inspection (psutil)
    4. Local weight file / directory inspection
    5. Analytical estimation based on parameter count & quantization
    """
    # 1. Backend direct inspection
    if backend_instance and hasattr(backend_instance, "get_model_ram_gb"):
        try:
            ram = backend_instance.get_model_ram_gb()
            if ram and ram > 0:
                logger.info(f"Auto-detected model memory from backend: [bold cyan]{ram} GB[/bold cyan]")
                return ram
        except Exception as e:
            logger.debug(f"Backend RAM query failed: {e}")

    # 2. Local weight file check
    if model_id:
        file_ram = detect_local_file_ram_gb(model_id)
        if file_ram and file_ram > 0.1:
            logger.info(f"Auto-detected model memory from local weights file: [bold cyan]{file_ram} GB[/bold cyan]")
            return file_ram

    # 3. OpenAI-compatible / Ollama endpoint probe
    if endpoint_url and model_id:
        ollama_ram = detect_ollama_model_ram_gb(endpoint_url, model_id)
        if ollama_ram and ollama_ram > 0:
            logger.info(f"Auto-detected model memory from endpoint telemetry: [bold cyan]{ollama_ram} GB[/bold cyan]")
            return ollama_ram

    # 4. Host runner process check
    proc_ram = detect_runner_process_ram_gb()
    if proc_ram and proc_ram > 0:
        logger.info(f"Auto-detected model memory from runner process: [bold cyan]{proc_ram} GB[/bold cyan]")
        return proc_ram

    # 5. Analytical parameter & quantization estimation
    est_ram = estimate_model_ram_gb(param_count_b, quantization)
    if est_ram:
        logger.info(f"Auto-estimated theoretical model RAM: [bold cyan]{est_ram} GB[/bold cyan] ({param_count_b}B @ {quantization})")
        return est_ram

    return None
