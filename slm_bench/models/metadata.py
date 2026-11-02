"""
Model metadata extraction, parameter count estimation, and classification.
"""

import re
import requests
from typing import Optional, Dict, Any
from .schema import ModelMetadata
from ..utils.logger import logger


def generate_model_slug(model_id: str, quantization: str) -> str:
    """Generate a clean, filesystem-safe and Git branch-safe model slug."""
    clean_id = model_id.strip().replace("\\", "/").split("/")[-1]
    # Replace non-alphanumeric (except dashes and dots)
    slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "-", f"{clean_id}-{quantization}")
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def determine_param_tier(param_count_b: Optional[float]) -> str:
    """Classify model into an SLM benchmark parameter tier."""
    if param_count_b is None:
        return "Unknown"
    if param_count_b < 3.0:
        return "<3B"
    elif param_count_b <= 8.5:
        return "3B-8B"
    elif param_count_b <= 15.0:
        return "8B-14B"
    elif param_count_b <= 35.0:
        return "14B-35B"
    elif param_count_b <= 50.0:
        return "35B-50B"
    else:
        return ">50B"


def extract_params_from_name(name: str) -> Optional[float]:
    """Parse parameter count from model name string (e.g., '7B', '3.8b', '14B', '0.5B')."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:-|\b|_|$)", name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # Check for million (e.g. 500M)
    m_match = re.search(r"(\d+)\s*[mM](?:-|\b|_|$)", name)
    if m_match:
        try:
            return round(float(m_match.group(1)) / 1000.0, 3)
        except ValueError:
            pass
    return None


def fetch_hf_model_metadata(
    model_id: str,
    quantization: str = "BF16",
    backend: str = "transformers",
    token: Optional[str] = None
) -> ModelMetadata:
    """
    Retrieve model metadata from Hugging Face API or infer from name/config.
    """
    logger.info(f"Fetching metadata for Hugging Face model: [bold cyan]{model_id}[/bold cyan]")
    headers = {"User-Agent": "slm-bench/0.1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    param_count_b: Optional[float] = extract_params_from_name(model_id)
    architecture: Optional[str] = None
    context_length: Optional[int] = None
    license_name: Optional[str] = None

    # Attempt query to Hugging Face Hub API
    api_url = f"https://huggingface.co/api/models/{model_id}"
    try:
        resp = requests.get(api_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Safetensors parameter count if reported by HF
            safetensors_info = data.get("safetensors", {})
            if isinstance(safetensors_info, dict) and "total" in safetensors_info:
                total_params = safetensors_info["total"]
                param_count_b = round(total_params / 1e9, 2)
            
            # Card tags
            tags = data.get("tags", [])
            for tag in tags:
                if tag.startswith("license:"):
                    license_name = tag.split("license:", 1)[1]

            # Model config from raw endpoint
            config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
            cfg_resp = requests.get(config_url, headers=headers, timeout=4)
            if cfg_resp.status_code == 200:
                cfg = cfg_resp.json()
                if "architectures" in cfg and cfg["architectures"]:
                    architecture = cfg["architectures"][0]
                elif "model_type" in cfg:
                    architecture = cfg["model_type"]

                # Context length
                for key in ["max_position_embeddings", "max_sequence_length", "seq_length", "context_length"]:
                    if key in cfg and isinstance(cfg[key], int):
                        context_length = cfg[key]
                        break
        else:
            logger.warning(f"Hugging Face API returned status {resp.status_code} for {model_id}. Using heuristic fallback.")
    except Exception as e:
        logger.warning(f"Could not reach Hugging Face Hub API ({e}). Using local heuristics.")

    # Default fallback for param_count if still unknown
    if param_count_b is None:
        param_count_b = 7.0  # Common default for SLMs if unknown

    tier = determine_param_tier(param_count_b)
    slug = generate_model_slug(model_id, quantization)

    return ModelMetadata(
        model_id=model_id,
        model_slug=slug,
        param_count_b=param_count_b,
        param_tier=tier,
        architecture=architecture or "CausalLM",
        quantization=quantization,
        context_length=context_length or 8192,
        license=license_name or "unknown",
        backend=backend,
    )
