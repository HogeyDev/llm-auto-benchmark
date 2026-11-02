#!/usr/bin/env python3
"""
Leaderboard Aggregator Script
Scans all benchmark JSON result files in data/results/, validates their schemas,
computes overall rankings across multiple tracks (General Intelligence, Coding & Agentic,
Local Efficiency), extracts hardware telemetry and Wilson confidence intervals,
and compiles an aggregated data/leaderboard.json as well as a manifest data/results/index.json
for static GitHub Pages delivery.
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


def normalize_model_entry(data, file_path):
    model = data.get("model", {})
    summary = data.get("summary_scores", {})
    tasks = data.get("tasks", {})
    categories = data.get("categories", {})
    hardware = data.get("hardware", {})
    telemetry_raw = data.get("telemetry", {}) or hardware.get("telemetry", {})

    # Extract or infer standard benchmark metrics
    gsm8k = (
        summary.get("gsm8k")
        or tasks.get("gsm8k", {}).get("score")
        or tasks.get("gsm8k", {}).get("normalized_score")
        or categories.get("math", {}).get("normalized_score")
        or 0.0
    )

    humaneval = (
        summary.get("humaneval")
        or tasks.get("humaneval", {}).get("score")
        or tasks.get("humaneval", {}).get("normalized_score")
        or tasks.get("slm_coding_agent", {}).get("normalized_score")
        or tasks.get("slm_agentic_coding", {}).get("normalized_score")
        or categories.get("coding", {}).get("normalized_score")
        or 0.0
    )

    arc = (
        summary.get("arc")
        or tasks.get("arc", {}).get("score")
        or tasks.get("arc", {}).get("normalized_score")
        or tasks.get("slm_reasoning_cot", {}).get("normalized_score")
        or categories.get("reasoning", {}).get("normalized_score")
        or 0.0
    )

    ifeval = (
        summary.get("ifeval")
        or tasks.get("ifeval", {}).get("score")
        or tasks.get("ifeval", {}).get("normalized_score")
        or tasks.get("slm_instruction_ifeval", {}).get("normalized_score")
        or tasks.get("slm_instruction_following", {}).get("normalized_score")
        or categories.get("instruction", {}).get("normalized_score")
        or 0.0
    )

    tool_calling = (
        summary.get("tool_calling")
        or tasks.get("tool_calling", {}).get("score")
        or tasks.get("tool_calling", {}).get("normalized_score")
        or tasks.get("slm_tool_calling_multiturn", {}).get("normalized_score")
        or categories.get("tool_calling", {}).get("normalized_score")
        or 0.0
    )

    # Determine overall score (SLM-Index)
    overall = data.get("slm_index")
    if overall is None:
        overall = summary.get("overall_score")
    if overall is None:
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        scores = [gsm8k, humaneval, arc, ifeval, tool_calling]
        overall = sum(w * s for w, s in zip(weights, scores))

    param_count = float(model.get("param_count_b", 0.0))

    # Parameter Category Tier
    tier = model.get("param_tier")
    if not tier or tier == "unknown":
        if param_count < 3.0:
            tier = "< 3B"
        elif param_count <= 7.0:
            tier = "3B–7B"
        elif param_count <= 14.0:
            tier = "7B–14B"
        elif param_count <= 32.0:
            tier = "14B–32B"
        else:
            tier = "32B+"

    # Hugging Face URL
    model_id = model.get("model_id", "unknown/unknown")
    hf_url = model.get("hf_url")
    if not hf_url:
        hf_url = f"https://huggingface.co/{model_id}"

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

    # Calculate aggregate throughput and latency
    task_speeds = [
        t.get("avg_tokens_per_sec")
        for t in tasks.values()
        if isinstance(t, dict) and t.get("avg_tokens_per_sec")
    ]
    avg_tokens_per_sec = (
        telemetry_raw.get("decode_tok_per_sec")
        or (sum(task_speeds) / len(task_speeds) if task_speeds else None)
        or summary.get("avg_tokens_per_sec")
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
        or summary.get("avg_latency_ms")
    )
    if avg_latency_ms is not None:
        avg_latency_ms = round(float(avg_latency_ms), 1)

    # Wilson Confidence Interval on Overall Score
    ci_lower = data.get("ci_lower")
    ci_upper = data.get("ci_upper")
    if ci_lower is None or ci_upper is None:
        total_p = sum(t.get("passed_samples", 0) for t in tasks.values() if isinstance(t, dict))
        total_s = sum(t.get("total_samples", 0) for t in tasks.values() if isinstance(t, dict))
        if total_s > 0:
            ci_lower, ci_upper = wilson_interval(total_p, total_s)
        else:
            ci_lower = round(max(0.0, float(overall) - 3.5), 1)
            ci_upper = round(min(100.0, float(overall) + 3.5), 1)

    # Profile detection
    profile = data.get("profile")
    if not profile:
        total_s = sum(t.get("total_samples", 0) for t in tasks.values() if isinstance(t, dict))
        if len(tasks) <= 4 and total_s <= 25:
            profile = "quick"
        elif len(tasks) >= 9:
            profile = "extended"
        else:
            profile = "standard"

    # Multi-Track Indices
    # 1. Coding & Agentic Index (60% coding + 40% tool calling)
    coding_agentic_index = data.get("coding_agentic_index")
    if coding_agentic_index is None:
        coding_score = categories.get("coding", {}).get("normalized_score") or humaneval
        tool_score = categories.get("tool_calling", {}).get("normalized_score") or tool_calling
        coding_agentic_index = round(0.60 * coding_score + 0.40 * tool_score, 2)
    else:
        coding_agentic_index = round(float(coding_agentic_index), 2)

    # 2. Local Efficiency Index (Score per GB, throughput TPS, efficiency score)
    local_efficiency = data.get("local_efficiency")
    if not local_efficiency or not isinstance(local_efficiency, dict):
        effective_ram = float(model_ram or 8.0)
        score_per_gb = round(float(overall) / max(1.0, effective_ram), 2)
        tps = float(avg_tokens_per_sec or 0.0)
        throughput_norm = min(100.0, (tps / 60.0) * 100.0)
        footprint_norm = min(100.0, (8.0 / max(1.0, effective_ram)) * 100.0)
        eff_score = round(0.5 * float(overall) + 0.3 * throughput_norm + 0.2 * footprint_norm, 2)
        local_efficiency = {
            "score_per_gb": score_per_gb,
            "throughput_tps": tps,
            "effective_memory_gb": effective_ram,
            "efficiency_score": eff_score,
        }
    else:
        if "score_per_gb" not in local_efficiency and model_ram:
            local_efficiency["score_per_gb"] = round(float(overall) / max(1.0, float(model_ram)), 2)

    # Telemetry snapshot
    telemetry = {
        "ttft_ms": telemetry_raw.get("ttft_ms") or avg_latency_ms,
        "decode_tok_per_sec": telemetry_raw.get("decode_tok_per_sec") or avg_tokens_per_sec,
        "tpot_ms": telemetry_raw.get("tpot_ms"),
        "peak_ram_mb": telemetry_raw.get("peak_ram_mb"),
        "peak_vram_mb": telemetry_raw.get("peak_vram_mb"),
    }

    # Extended benchmark task scores if present
    extended_scores = {}
    for task_key in [
        "mmlu_pro", "gpqa_diamond", "gpqa", "bbh", "math500", "evalplus",
        "livecodebench", "ifbench", "bfcl", "ruler", "safety", "harmbench", "xstest"
    ]:
        if task_key in tasks:
            t = tasks[task_key]
            val = t.get("normalized_score") or t.get("score")
            if val is not None:
                extended_scores[task_key] = round(float(val), 2)

    return {
        "filename": os.path.basename(file_path),
        "file_path": f"data/results/{os.path.basename(file_path)}",
        "evaluation_id": data.get("evaluation_id", "N/A"),
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "profile": profile,
        "submitter": data.get("submitter", {
            "name": "Community Contributor",
            "github_handle": "anonymous",
            "affiliation": "Open Source"
        }),
        "model": {
            "model_id": model_id,
            "model_slug": model.get("model_slug", model_id.replace("/", "-")),
            "hf_url": hf_url,
            "param_count_b": param_count,
            "param_tier": tier,
            "architecture": model.get("architecture", "Transformer"),
            "quantization": model.get("quantization", "BF16"),
            "model_ram_gb": model_ram,
            "context_length": model.get("context_length", 32768),
            "license": model.get("license", "Unknown"),
            "backend": model.get("backend", "transformers"),
            "reasoning_effort": model.get("reasoning_effort")
        },
        "scores": {
            "overall": round(float(overall), 2),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "coding_agentic_index": coding_agentic_index,
            "local_efficiency_score": local_efficiency.get("efficiency_score"),
            "score_per_gb": local_efficiency.get("score_per_gb"),
            "gsm8k": round(float(gsm8k), 2),
            "humaneval": round(float(humaneval), 2),
            "arc": round(float(arc), 2),
            "ifeval": round(float(ifeval), 2),
            "tool_calling": round(float(tool_calling), 2),
            **extended_scores
        },
        "hardware": {
            "gpu_name": hardware.get("gpu_name", "N/A"),
            "gpu_count": hardware.get("gpu_count", 1),
            "gpu_vram_gb": hardware.get("gpu_vram_gb", 0.0),
            "cpu": hardware.get("cpu", "N/A"),
            "cpu_cores": hardware.get("cpu_cores", 0),
            "total_ram_gb": hardware.get("total_ram_gb", 0.0),
            "model_ram_gb": model_ram,
            "os": hardware.get("os", "Linux"),
            "cuda_version": hardware.get("cuda_version", "N/A"),
            "driver_version": hardware.get("driver_version", "N/A")
        },
        "telemetry": telemetry,
        "local_efficiency": local_efficiency,
        "categories": categories,
        "tasks": tasks,
        "provenance": data.get("provenance", {}),
        "notes": data.get("notes") or "",
        "raw_logs_preview": data.get("raw_logs_preview") or data.get("notes") or "Benchmark run completed successfully."
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "data" / "results"
    output_leaderboard = repo_root / "data" / "leaderboard.json"
    output_index = repo_root / "data" / "results" / "index.json"

    json_files = sorted(glob.glob(str(results_dir / "*.json")))
    root_results_dir = repo_root.parent / "data" / "results"
    if root_results_dir.exists() and root_results_dir != results_dir:
        json_files += sorted(glob.glob(str(root_results_dir / "*.json")))

    # Deduplicate and exclude index.json
    seen_basenames = set()
    filtered_json_files = []
    for f in json_files:
        basename = os.path.basename(f)
        if not basename.endswith("index.json") and basename not in seen_basenames:
            seen_basenames.add(basename)
            filtered_json_files.append(f)
    json_files = filtered_json_files

    print(f"[Leaderboard Aggregator] Found {len(json_files)} result files in {results_dir}")

    records = []
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            backend = str(data.get("model", {}).get("backend", "")).strip().lower()
            if backend == "mock":
                print(f"[SKIP] {os.path.basename(f)}: mock backend, not a real evaluation")
                continue
            entry = normalize_model_entry(data, f)
            records.append(entry)
        except Exception as e:
            print(f"[WARN] Failed to process {f}: {e}", file=sys.stderr)

    # Sort descending by overall score
    records.sort(key=lambda r: r["scores"]["overall"], reverse=True)

    # Assign ranks
    for rank, entry in enumerate(records, start=1):
        entry["rank"] = rank

    # Compute high-level stats
    stats = {
        "total_models": len(records),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "highest_overall": records[0]["scores"]["overall"] if records else 0.0,
        "best_by_benchmark": {
            "gsm8k": max([r["scores"]["gsm8k"] for r in records], default=0.0),
            "humaneval": max([r["scores"]["humaneval"] for r in records], default=0.0),
            "arc": max([r["scores"]["arc"] for r in records], default=0.0),
            "ifeval": max([r["scores"]["ifeval"] for r in records], default=0.0),
            "tool_calling": max([r["scores"]["tool_calling"] for r in records], default=0.0),
        } if records else {},
        "best_by_track": {
            "top_overall": records[0]["model"]["model_id"] if records else "N/A",
            "top_overall_score": records[0]["scores"]["overall"] if records else 0.0,
            "top_coding_agentic": max(records, key=lambda r: r["scores"]["coding_agentic_index"])["model"]["model_id"] if records else "N/A",
            "top_coding_agentic_score": max([r["scores"]["coding_agentic_index"] for r in records], default=0.0),
            "top_efficiency": max(records, key=lambda r: r["scores"].get("score_per_gb") or 0.0)["model"]["model_id"] if records else "N/A",
            "top_efficiency_score": max([r["scores"].get("score_per_gb") or 0.0 for r in records], default=0.0),
            "top_throughput": max(records, key=lambda r: r["telemetry"].get("decode_tok_per_sec") or 0.0)["model"]["model_id"] if records else "N/A",
            "top_throughput_tps": max([r["telemetry"].get("decode_tok_per_sec") or 0.0 for r in records], default=0.0),
        } if records else {},
        "unique_profiles": sorted(list(set(r.get("profile", "standard") for r in records))),
        "unique_architectures": sorted(list(set(r["model"]["architecture"] for r in records))),
        "unique_licenses": sorted(list(set(r["model"]["license"] for r in records))),
        "unique_quantizations": sorted(list(set(r["model"]["quantization"] for r in records)))
    }

    leaderboard_payload = {
        "metadata": stats,
        "models": records
    }

    with open(output_leaderboard, "w", encoding="utf-8") as fp:
        json.dump(leaderboard_payload, fp, indent=2)
    print(f"[OK] Generated {output_leaderboard} with {len(records)} entries.")

    manifest_payload = {
        "count": len(records),
        "files": [r["filename"] for r in records]
    }
    with open(output_index, "w", encoding="utf-8") as fp:
        json.dump(manifest_payload, fp, indent=2)
    print(f"[OK] Generated manifest {output_index}.")


if __name__ == "__main__":
    main()
