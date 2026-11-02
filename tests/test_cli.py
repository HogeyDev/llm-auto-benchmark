"""
Tests for CLI commands using typer.testing.CliRunner.
"""

from typer.testing import CliRunner
from slm_bench.cli import app
from pathlib import Path

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Automated Evaluation Harness" in result.output


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Host Hardware & Runtime Environment" in result.output


def test_cli_evaluate_and_threshold(tmp_path):
    out_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "evaluate",
        "--model-id", "test/tiny-model-1B",
        "--backend", "mock",
        "--output-dir", str(out_dir),
        "--sample-limit", "1",
    ])
    assert result.exit_code == 0
    assert "Composite SLM-Index" in result.output
    # Check that file exists
    saved_files = list(out_dir.glob("*.json"))
    assert len(saved_files) == 1


def test_cli_over_50b_rejection(tmp_path):
    out_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "evaluate",
        "--model-id", "huge-model-70B",
        "--backend", "mock",
        "--output-dir", str(out_dir),
    ])
    # Should reject models over 50B
    assert result.exit_code != 0
    assert "exceeding the 50B SLM limit" in result.output


def test_cli_batch(tmp_path):
    batch_file = tmp_path / "models.txt"
    batch_file.write_text("model-a-3B\nmodel-b-7B\n", encoding="utf-8")
    out_dir = tmp_path / "results"

    result = runner.invoke(app, [
        "batch",
        "--batch-file", str(batch_file),
        "--backend", "mock",
        "--output-dir", str(out_dir),
        "--sample-limit", "1",
    ])
    assert result.exit_code == 0
    assert "Batch run completed for 2 models" in result.output
