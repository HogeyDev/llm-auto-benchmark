"""
Quantization Tradeoff Analysis Module.
Computes capability retention, accuracy degradation, memory reduction,
and speedup across quantization formats (FP16/BF16, Q8, Q6, Q5, Q4, Q3, INT4).
"""

from typing import Dict, List, Any, Optional
from ..models.schema import BenchmarkResult


def compare_quantizations(
    baseline: BenchmarkResult,
    quantized: BenchmarkResult,
) -> Dict[str, Any]:
    """
    Compare a quantized model evaluation against its higher-precision baseline.
    
    Returns:
        Dict containing accuracy retention, degradation %, RAM savings %, and speedup.
    """
    base_score = baseline.slm_index
    quant_score = quantized.slm_index
    score_delta = round(quant_score - base_score, 2)
    degradation_pct = round(((base_score - quant_score) / max(0.01, base_score)) * 100.0, 2)

    base_ram = baseline.hardware.model_ram_gb or baseline.model.model_ram_gb or 0.0
    quant_ram = quantized.hardware.model_ram_gb or quantized.model.model_ram_gb or 0.0
    ram_savings_gb = round(max(0.0, base_ram - quant_ram), 2)
    ram_savings_pct = round(((base_ram - quant_ram) / max(0.01, base_ram)) * 100.0, 2) if base_ram > 0 else 0.0

    return {
        "model_id": quantized.model.model_id,
        "baseline_quantization": baseline.model.quantization,
        "quantized_format": quantized.model.quantization,
        "baseline_slm_index": base_score,
        "quantized_slm_index": quant_score,
        "score_delta": score_delta,
        "degradation_pct": degradation_pct,
        "retention_pct": round(100.0 - degradation_pct, 2),
        "baseline_ram_gb": base_ram,
        "quantized_ram_gb": quant_ram,
        "ram_savings_gb": ram_savings_gb,
        "ram_savings_pct": ram_savings_pct,
    }
