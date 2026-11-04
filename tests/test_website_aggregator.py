"""
Unit tests for the Website leaderboard aggregator and static dataset compilation.
"""

import json
import shutil
import sys
from pathlib import Path

# Add scripts directory to sys.path
scripts_dir = Path(__file__).resolve().parent.parent / "Website" / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from aggregate import normalize_model_entry, wilson_interval, main as run_aggregator


def test_wilson_interval():
    low, high = wilson_interval(90, 100)
    assert 82.0 <= low <= 85.0
    assert 94.0 <= high <= 97.0
    assert low < high

    # Edge cases
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(0, 10) == (0.0, 27.75)
    assert wilson_interval(10, 10) == (72.25, 100.0)


def test_normalize_model_entry():
    sample_data = {
        "model": {
            "model_id": "test/sample-model-7b",
            "model_slug": "sample-model-7b",
            "param_count_b": 7.0,
            "quantization": "BF16",
            "license": "MIT",
            "backend": "vllm"
        },
        "hardware": {
            "gpu_name": "RTX 4090",
            "gpu_vram_gb": 24.0,
            "total_ram_gb": 64.0
        },
        "profile": "standard",
        "slm_index": 82.5,
        "coding_agentic_index": 81.0,
        "local_efficiency": {
            "score_per_gb": 12.5,
            "throughput_tps": 55.0,
            "effective_memory_gb": 6.6
        },
        "tasks": {
            "gsm8k": {"score": 85.0, "passed_samples": 85, "total_samples": 100},
            "humaneval": {"score": 80.0, "passed_samples": 80, "total_samples": 100}
        }
    }

    entry = normalize_model_entry(sample_data, "data/results/test_model.json")
    assert entry["model"]["model_id"] == "test/sample-model-7b"
    assert entry["profile"] == "standard"
    assert entry["scores"]["overall"] == 82.5
    assert entry["scores"]["coding_agentic_index"] == 81.0
    assert entry["scores"]["score_per_gb"] == 12.5
    assert entry["scores"]["ci_lower"] is not None
    assert entry["scores"]["ci_upper"] is not None
    assert entry["scores"]["ci_lower"] < entry["scores"]["ci_upper"]


def test_aggregator_main(tmp_path):
    """Run the aggregator against an isolated Website copy without touching tracked data."""
    scripts_dir = Path(__file__).resolve().parent.parent / "Website" / "scripts"
    website_dir = tmp_path / "Website"
    copied_scripts = website_dir / "scripts"
    results_dir = website_dir / "data" / "results"
    copied_scripts.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    shutil.copy2(scripts_dir / "aggregate.py", copied_scripts / "aggregate.py")
    sample_result = {
        "model": {
            "model_id": "test/sample-model-7b",
            "model_slug": "sample-model-7b",
            "param_count_b": 7.0,
            "param_tier": "3B-8B",
            "quantization": "BF16",
            "backend": "transformers",
            "license": "MIT",
        },
        "hardware": {
            "os": "TestOS",
            "cpu": "Test CPU",
            "cpu_cores": 8,
            "total_ram_gb": 32.0,
            "gpu_name": "Test GPU",
            "gpu_count": 1,
            "gpu_vram_gb": 24.0,
        },
        "profile": "standard",
        "status": "FULL",
        "eligible_for_leaderboard": True,
        "slm_index": 82.5,
        "coding_agentic_index": 81.0,
        "local_efficiency": {"score_per_gb": 12.5},
        "tasks": {
            "gsm8k": {
                "normalized_score": 85.0,
                "passed_samples": 85,
                "total_samples": 100,
            },
        },
        "categories": {
            "reasoning": {"normalized_score": 80.0},
            "mathematics": {"normalized_score": 85.0},
            "instruction": {"normalized_score": 82.0},
        },
    }
    (results_dir / "sample-model-7b.json").write_text(
        json.dumps(sample_result),
        encoding="utf-8",
    )

    import aggregate

    original_file = aggregate.__file__
    aggregate.__file__ = str(copied_scripts / "aggregate.py")
    try:
        run_aggregator()
    finally:
        aggregate.__file__ = original_file

    leaderboard_file = website_dir / "data" / "leaderboard.json"
    index_file = results_dir / "index.json"

    assert leaderboard_file.exists()
    assert index_file.exists()

    with open(leaderboard_file, "r", encoding="utf-8") as fp:
        lb = json.load(fp)

    assert lb["metadata"]["total_models"] == 1
    assert lb["metadata"]["official_models_count"] == 1
    assert lb["models"][0]["eligible_for_leaderboard"] is True
    assert lb["models"][0]["rank"] == 1
