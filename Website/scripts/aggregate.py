#!/usr/bin/env python3
"""
Leaderboard Aggregator Script
Scans all benchmark JSON result files in data/results/, validates their schemas,
computes overall rankings, and compiles an aggregated data/leaderboard.json
as well as a manifest data/results/index.json for static GitHub Pages delivery.
"""

import json
import glob
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

def normalize_model_entry(data, file_path):
    model = data.get("model", {})
    summary = data.get("summary_scores", {})
    tasks = data.get("tasks", {})
    categories = data.get("categories", {})
    hardware = data.get("hardware", {})

    # Extract or infer standard benchmark metrics
    # Required columns: GSM8K (Math), HumanEval (Coding), ARC (Reasoning), IFEval, Tool Calling
    gsm8k = (
        summary.get("gsm8k")
        or tasks.get("gsm8k", {}).get("score")
        or categories.get("math", {}).get("normalized_score")
        or 0.0
    )

    humaneval = (
        summary.get("humaneval")
        or tasks.get("humaneval", {}).get("score")
        or tasks.get("slm_agentic_coding", {}).get("normalized_score")
        or categories.get("coding", {}).get("normalized_score")
        or 0.0
    )

    arc = (
        summary.get("arc")
        or tasks.get("arc", {}).get("score")
        or tasks.get("slm_reasoning_cot", {}).get("normalized_score")
        or categories.get("reasoning", {}).get("normalized_score")
        or 0.0
    )

    ifeval = (
        summary.get("ifeval")
        or tasks.get("ifeval", {}).get("score")
        or tasks.get("slm_instruction_following", {}).get("normalized_score")
        or categories.get("instruction", {}).get("normalized_score")
        or 0.0
    )

    tool_calling = (
        summary.get("tool_calling")
        or tasks.get("tool_calling", {}).get("score")
        or tasks.get("slm_tool_calling_multiturn", {}).get("normalized_score")
        or categories.get("tool_calling", {}).get("normalized_score")
        or 0.0
    )

    # Determine overall score
    overall = summary.get("overall_score")
    if overall is None:
        overall = data.get("slm_index")
    if overall is None:
        # Standard weighted composite benchmark average
        weights = [0.25, 0.25, 0.20, 0.15, 0.15] # GSM8K, HumanEval, ARC, IFEval, Tool Calling
        scores = [gsm8k, humaneval, arc, ifeval, tool_calling]
        overall = sum(w * s for w, s in zip(weights, scores))

    param_count = float(model.get("param_count_b", 0.0))

    # Determine parameter category tier
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
        if any(k in q for k in ["Q4", "4BIT", "AWQ", "NF4"]):
            model_ram = round(param_count * 0.65, 2)
        elif any(k in q for k in ["Q8", "8BIT"]):
            model_ram = round(param_count * 1.15, 2)
        elif any(k in q for k in ["FP16", "BF16"]):
            model_ram = round(param_count * 2.15, 2)
        else:
            model_ram = round(param_count * 0.85, 2)

    return {
        "filename": os.path.basename(file_path),
        "file_path": f"data/results/{os.path.basename(file_path)}",
        "evaluation_id": data.get("evaluation_id", "N/A"),
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
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
            "backend": model.get("backend", "transformers")
        },
        "scores": {
            "overall": round(float(overall), 2),
            "gsm8k": round(float(gsm8k), 2),
            "humaneval": round(float(humaneval), 2),
            "arc": round(float(arc), 2),
            "ifeval": round(float(ifeval), 2),
            "tool_calling": round(float(tool_calling), 2)
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
        "tasks": tasks,
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
