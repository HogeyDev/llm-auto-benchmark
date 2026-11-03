#!/usr/bin/env python3
"""
Leaderboard Aggregator Script (Version 2.0).
Scans benchmark JSON result files in data/results/, validates their schemas,
strictly segregates deprecated custom SLM benchmarks from official benchmarks,
extracts Wilson 95% confidence intervals, hardware telemetry, and status flags (FULL vs PARTIAL),
and compiles aggregated data/leaderboard.json and data/results/index.json for GitHub Pages.
"""

import json
import glob
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def wilson_interval(passed: int, total: int, z: float = 1.95996) -> tuple[float, float]:
    """Calculate 95% Wilson score interval for passed/total binomial proportion (0-100%)."""
    if total <= 0:
        return 0.0, 0.0
    p = passed / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z**2) / (4.0 * total**2)))
    lower = max(0.0, (center - margin) * 100.0)
    upper = min(100.0, (center + margin) * 100.0)
    return round(lower, 2), round(upper, 2)


def is_legacy_custom_slm_result(data: dict) -> bool:
    """Detect if a result file is from the deprecated custom handcrafted SLM benchmarks."""
    tasks = data.get("tasks", {})
    legacy_keys = {
        "slm_reasoning_cot",
        "slm_coding_agent",
        "slm_instruction_ifeval",
        "slm_tool_calling_multiturn",
        "slm_agentic_coding",
        "slm_instruction_following",
    }
    return any(k in tasks for k in legacy_keys) or data.get("schema_version") == "1.0.0" and not any(
        k in tasks for k in ["mmlu_pro", "gpqa_diamond", "math500"]
    )


def normalize_model_entry(data: dict, file_path: str) -> dict:
    model = data.get("model", {})
    tasks = data.get("tasks", {})
    categories = data.get("categories", {})
    hardware = data.get("hardware", {})
    telemetry_raw = data.get("telemetry", {}) or hardware.get("telemetry", {})

    is_legacy = is_legacy_custom_slm_result(data) or data.get("is_legacy", False)
    status = "INVALID" if is_legacy else data.get("status", "FULL")
    eligible = False if is_legacy else data.get("eligible_for_leaderboard", (status == "FULL"))

    # Extract official benchmark scores
    mmlu_pro = tasks.get("mmlu_pro", {}).get("normalized_score")
    gpqa = tasks.get("gpqa_diamond", {}).get("normalized_score") or tasks.get("gpqa", {}).get("normalized_score")
    bbh = tasks.get("bbh", {}).get("normalized_score")
    gsm8k = tasks.get("gsm8k", {}).get("normalized_score")
    math500 = tasks.get("math500", {}).get("normalized_score")
    ifeval = tasks.get("ifeval", {}).get("normalized_score")
    evalplus = tasks.get("evalplus", {}).get("normalized_score")
    bfcl = tasks.get("bfcl", {}).get("normalized_score")
    xstest = tasks.get("xstest", {}).get("normalized_score") or tasks.get("safety", {}).get("normalized_score")

    # Categories
    reasoning_cat = categories.get("reasoning", {}).get("normalized_score")
    math_cat = categories.get("mathematics", {}).get("normalized_score")
    instruction_cat = categories.get("instruction", {}).get("normalized_score")
    coding_cat = categories.get("coding", {}).get("normalized_score")
    tool_cat = categories.get("tool_calling", {}).get("normalized_score")

    slm_index = data.get("slm_index", 0.0)
    param_count = float(model.get("param_count_b", 0.0) or 0.0)

    # Parameter Category Tier
    tier = model.get("param_tier")
    if not tier or tier == "unknown":
        if param_count < 3.0:
            tier = "< 3B"
        elif param_count <= 8.0:
            tier = "3B–8B"
        elif param_count <= 14.0:
            tier = "8B–14B"
        elif param_count <= 35.0:
            tier = "14B–35B"
        else:
            tier = "35B+"

    model_id = model.get("model_id", "unknown/unknown")
    hf_url = model.get("hf_url") or f"https://huggingface.co/{model_id}"

    model_ram = hardware.get("model_ram_gb") or model.get("model_ram_gb") or data.get("model_ram_gb")
    if model_ram is None and param_count:
        q = (model.get("quantization") or "").upper()
        if any(k in q for k in ["Q4", "4BIT", "AWQ", "NF4", "INT4"]):
            model_ram = round(param_count * 0.65, 2)
        elif any(k in q for k in ["Q8", "8BIT", "INT8"]):
            model_ram = round(param_count * 1.15, 2)
        elif any(k in q for k in ["FP16", "BF16"]):
            model_ram = round(param_count * 2.15, 2)
        else:
            model_ram = round(param_count * 0.85, 2)

    task_speeds = [
        t.get("avg_tokens_per_sec")
        for t in tasks.values()
        if isinstance(t, dict) and t.get("avg_tokens_per_sec")
    ]
    avg_tokens_per_sec = (
        telemetry_raw.get("decode_tok_per_sec")
        or (sum(task_speeds) / len(task_speeds) if task_speeds else None)
    )
    if avg_tokens_per_sec is not None:
        avg_tokens_per_sec = round(float(avg_tokens_per_sec), 1)

    task_latencies = [
        t.get("avg_latency_ms")
        for t in tasks.values()
        if isinstance(t, dict) and t.get("avg_latency_ms")
    ]
    avg_latency_ms = (
        telemetry_raw.get("ttft_ms")
        or (sum(task_latencies) / len(task_latencies) if task_latencies else None)
    )
    if avg_latency_ms is not None:
        avg_latency_ms = round(float(avg_latency_ms), 1)

    # Wilson CI calculation across evaluated samples
    total_p = sum(t.get("passed_samples", 0) for t in tasks.values() if isinstance(t, dict))
    total_s = sum(t.get("total_samples", 0) for t in tasks.values() if isinstance(t, dict))
    if total_s > 0:
        ci_lower, ci_upper = wilson_interval(total_p, total_s)
    else:
        ci_lower, ci_upper = 0.0, 0.0

    profile = data.get("profile", "standard")

    return {
        "filename": os.path.basename(file_path),
        "file_path": f"data/results/{os.path.basename(file_path)}",
        "evaluation_id": data.get("evaluation_id", "N/A"),
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "profile": profile,
        "status": status,
        "eligible_for_leaderboard": eligible,
        "is_legacy": is_legacy,
        "model": {
            "model_id": model_id,
            "model_slug": model.get("model_slug", model_id.replace("/", "-")),
            "hf_url": hf_url,
            "param_count_b": param_count,
            "param_tier": tier,
            "architecture": model.get("architecture", "Transformer"),
            "quantization": model.get("quantization", "BF16"),
            "model_ram_gb": model_ram,
            "backend": model.get("backend", "transformers"),
            "license": model.get("license", "Unknown"),
            "reasoning_effort": model.get("reasoning_effort"),
        },
        "scores": {
            "slm_index": round(float(slm_index), 2),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "coding_agentic": data.get("coding_agentic_index"),
            # Canonical Official Benchmarks
            "mmlu_pro": round(float(mmlu_pro), 2) if mmlu_pro is not None else None,
            "gpqa_diamond": round(float(gpqa), 2) if gpqa is not None else None,
            "bbh": round(float(bbh), 2) if bbh is not None else None,
            "gsm8k": round(float(gsm8k), 2) if gsm8k is not None else None,
            "math500": round(float(math500), 2) if math500 is not None else None,
            "ifeval": round(float(ifeval), 2) if ifeval is not None else None,
            "evalplus": round(float(evalplus), 2) if evalplus is not None else None,
            "bfcl": round(float(bfcl), 2) if bfcl is not None else None,
            "xstest": round(float(xstest), 2) if xstest is not None else None,
            # Compatibility aliases
            "overall": round(float(slm_index), 2),
            "coding_agentic_index": data.get("coding_agentic_index") or data.get("scores", {}).get("coding_agentic_index"),
            "score_per_gb": (data.get("local_efficiency") or {}).get("score_per_gb") if isinstance(data.get("local_efficiency"), dict) else None,
            # Pillar Aggregates
            "reasoning": round(float(reasoning_cat), 2) if reasoning_cat is not None else None,
            "mathematics": round(float(math_cat), 2) if math_cat is not None else None,
            "instruction": round(float(instruction_cat), 2) if instruction_cat is not None else None,
            "coding": round(float(coding_cat), 2) if coding_cat is not None else None,
            "tool_calling": round(float(tool_cat), 2) if tool_cat is not None else None,
        },
        "hardware": {
            "os": hardware.get("os", "Linux"),
            "cpu": hardware.get("cpu", "Unknown CPU"),
            "cpu_cores": hardware.get("cpu_cores", 8),
            "total_ram_gb": hardware.get("total_ram_gb", 16.0),
            "gpu_name": hardware.get("gpu_name") or "CPU",
            "gpu_count": hardware.get("gpu_count", 0),
            "gpu_vram_gb": hardware.get("gpu_vram_gb"),
            "cuda_version": hardware.get("cuda_version"),
        },
        "telemetry": {
            "ttft_ms": avg_latency_ms,
            "decode_tok_per_sec": avg_tokens_per_sec,
            "total_samples": total_s,
            "passed_samples": total_p,
        },
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    results_dirs = [
        base_dir / "data" / "results",
        Path("data/results"),
        Path("Website/data/results"),
        Path("website/data/results"),
    ]
    results_dir = None
    for d in results_dirs:
        if d.exists():
            results_dir = d
            break

    if not results_dir:
        print("No results directory found.")
        sys.exit(0)

    json_files = glob.glob(str(results_dir / "*.json"))
    models = []
    for f in json_files:
        if f.endswith("index.json") or f.endswith("leaderboard.json"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            entry = normalize_model_entry(data, f)
            models.append(entry)
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    # Sort: Eligible full official runs first, by SLM-Index descending
    models.sort(key=lambda m: (1 if m["eligible_for_leaderboard"] else 0, m["scores"]["slm_index"]), reverse=True)

    # Assign ranks to eligible official runs
    rank = 1
    for m in models:
        if m["eligible_for_leaderboard"]:
            m["rank"] = rank
            rank += 1
        else:
            m["rank"] = "-"

    metadata = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_models": len(models),
        "official_models_count": sum(1 for m in models if m["eligible_for_leaderboard"]),
    }

    leaderboard = {
        "metadata": metadata,
        "updated_at": metadata["updated_at"],
        "total_models": len(models),
        "official_models_count": metadata["official_models_count"],
        "models": models,
    }

    out_file = results_dir.parent / "leaderboard.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(leaderboard, fp, indent=2)

    index_data = {
        "updated_at": metadata["updated_at"],
        "total_models": len(models),
        "models": [
            {
                "filename": m["filename"],
                "model_id": m["model"]["model_id"],
                "model_slug": m["model"]["model_slug"],
                "slm_index": m["scores"]["slm_index"],
                "status": m["status"],
                "eligible": m["eligible_for_leaderboard"],
            }
            for m in models
        ],
    }
    with open(results_dir / "index.json", "w", encoding="utf-8") as fp:
        json.dump(index_data, fp, indent=2)

    print(f"Successfully aggregated {len(models)} models ({leaderboard['official_models_count']} official) into {out_file}")


if __name__ == "__main__":
    main()
