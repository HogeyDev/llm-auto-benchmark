"""
CLI interface for SLM-Bench evaluation harness and submission system.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List
import typer
from rich.table import Table
from rich.panel import Panel

from .models.hardware import get_hardware_info, generate_auto_system_notes
from .models.metadata import fetch_hf_model_metadata
from .models.memory_detector import auto_detect_model_ram_gb
from .models.schema import BenchmarkResult
from .backends.registry import get_backend
from .benchmarks.suite import BenchmarkSuite
from .submission.git_manager import GitSubmissionManager
from .submission.pr_manager import PullRequestManager
from .utils.logger import console, logger, setup_logger

app = typer.Typer(
    name="slm-bench",
    help="Automated Evaluation Harness and GitHub PR Submission System for Small Local AI.",
    add_completion=False,
)


@app.command()
def info(
    model_id: Optional[str] = typer.Option(None, "--model-id", "-m", help="Hugging Face model repository ID to inspect"),
    quantization: str = typer.Option("BF16", "--quantization", "-q", help="Quantization tag"),
    model_ram_gb: Optional[float] = typer.Option(None, "--model-ram-gb", "--ram-gb", help="Size in RAM / VRAM in GB that the model takes up"),
):
    """Display host hardware specifications and inspect Hugging Face model metadata."""
    setup_logger()
    hw = get_hardware_info(model_ram_gb=model_ram_gb)

    hw_table = Table(title="[Host Hardware & Runtime Environment]", show_header=True, header_style="bold magenta")
    hw_table.add_column("Property", style="dim")
    hw_table.add_column("Value", style="bold cyan")
    hw_table.add_row("Operating System", hw.os)
    hw_table.add_row("CPU", f"{hw.cpu} ({hw.cpu_cores} logical cores)")
    hw_table.add_row("System RAM", f"{hw.total_ram_gb} GB")
    if hw.model_ram_gb:
        hw_table.add_row("Model RAM Footprint", f"{hw.model_ram_gb} GB")
    hw_table.add_row("GPU Model", hw.gpu_name or "None detected (CPU mode)")
    hw_table.add_row("GPU VRAM", f"{hw.gpu_vram_gb} GB" if hw.gpu_vram_gb else "N/A")
    hw_table.add_row("CUDA Version", hw.cuda_version or "N/A")
    hw_table.add_row("Driver Version", hw.driver_version or "N/A")
    console.print(hw_table)

    if model_id:
        meta = fetch_hf_model_metadata(model_id, quantization=quantization, model_ram_gb=model_ram_gb)
        m_table = Table(title=f"[Model Metadata: {model_id}]", show_header=True, header_style="bold green")
        m_table.add_column("Field", style="dim")
        m_table.add_column("Value", style="bold yellow")
        m_table.add_row("Model Slug", meta.model_slug)
        m_table.add_row("Parameter Count", f"{meta.param_count_b:.2f}B" if meta.param_count_b else "Unknown")
        m_table.add_row("Parameter Tier", meta.param_tier)
        m_table.add_row("RAM / VRAM Footprint", f"{meta.model_ram_gb:.2f} GB" if meta.model_ram_gb else "Unknown")
        m_table.add_row("Architecture", meta.architecture or "Unknown")
        m_table.add_row("Context Window", f"{meta.context_length} tokens" if meta.context_length else "Unknown")
        m_table.add_row("License", meta.license or "Unknown")
        console.print(m_table)


@app.command()
def evaluate(
    model_id: str = typer.Option(..., "--model-id", "--repo-id", "-m", help="Hugging Face repo ID or local model path"),
    quantization: str = typer.Option("BF16", "--quantization", "-q", help="Quantization: BF16, FP16, AWQ_4BIT, GGUF_Q4_K_M, NF4"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Inference backend: mock, transformers, llamacpp, vllm, openai (Ollama / LM Studio)"),
    endpoint_url: str = typer.Option("http://localhost:11434/v1", "--endpoint-url", "--endpoint", "--base-url", help="OpenAI-compatible server endpoint URL (e.g. http://localhost:11434/v1, http://localhost:1234/v1, or custom runner)"),
    api_key: str = typer.Option("EMPTY", "--api-key", envvar="OPENAI_API_KEY", help="API key for custom or remote endpoint if required"),
    model_ram_gb: Optional[float] = typer.Option(None, "--model-ram-gb", "--ram-gb", help="Size in RAM / VRAM in GB that the model takes up"),
    output_dir: str = typer.Option("data/results", "--output-dir", "-o", help="Target directory for output JSON"),
    sample_limit: Optional[int] = typer.Option(None, "--sample-limit", "-n", help="Limit samples per task (for testing)"),
    tasks: Optional[str] = typer.Option(None, "--tasks", "-t", help="Comma-separated list of tasks to run (reasoning, coding, instruction, tool_calling)"),
    submit: bool = typer.Option(False, "--submit", "-s", help="Automatically submit results via Pull Request after eval"),
    github_repo: Optional[str] = typer.Option(None, "--github-repo", help="Target GitHub repository (e.g. owner/repo)"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN", help="GitHub Personal Access Token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode: do not push Git commits or open real PRs"),
    allow_over_50b: bool = typer.Option(True, "--allow-over-50b", help="Deprecated flag maintained for backwards compatibility"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Optional notes on runner configuration, offloading, or execution details"),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort", "--effort", help="Reasoning effort level for reasoning models (e.g. low, medium, high)"),
):
    """Run benchmark evaluation suite on a single model and save standardized JSON."""
    setup_logger()
    if isinstance(reasoning_effort, typer.models.OptionInfo):
        reasoning_effort = None
    logger.info(f"Initiating evaluation for [bold green]{model_id}[/bold green] using [cyan]{backend}[/cyan] backend...")

    # 1. Hardware and Model Metadata
    meta = fetch_hf_model_metadata(model_id, quantization=quantization, backend=backend, model_ram_gb=model_ram_gb)
    if reasoning_effort:
        meta.reasoning_effort = reasoning_effort
    hw = get_hardware_info(model_ram_gb=meta.model_ram_gb)

    # 2. Model Telemetry Logging (Small Local AI)
    if meta.param_count_b:
        logger.info(f"Model '{model_id}' evaluated at {meta.param_count_b:.1f}B parameters (class: Small Local AI).")
    if meta.model_ram_gb:
        logger.info(f"Model RAM / VRAM footprint: [bold cyan]{meta.model_ram_gb:.2f} GB[/bold cyan]")
    if reasoning_effort:
        logger.info(f"Reasoning Effort: [bold yellow]{reasoning_effort}[/bold yellow]")

    # 3. Instantiate Backend
    try:
        engine = get_backend(
            backend,
            model_id=model_id,
            quantization=quantization,
            endpoint_url=endpoint_url,
            api_key=api_key,
            model_ram_gb=meta.model_ram_gb,
            reasoning_effort=reasoning_effort,
        )
        engine.load()
    except Exception as e:
        logger.error(f"Failed to initialize backend '{backend}': {e}")
        raise typer.Exit(code=1)

    # 4. Auto-detect Model RAM / VRAM if not explicitly specified
    if model_ram_gb is None:
        detected_ram = auto_detect_model_ram_gb(
            backend_instance=engine,
            backend_name=backend,
            model_id=model_id,
            quantization=quantization,
            param_count_b=meta.param_count_b,
            endpoint_url=endpoint_url if backend in ("openai", "ollama", "lmstudio", "endpoint", "localai") else None,
        )
        if detected_ram:
            meta.model_ram_gb = detected_ram
            hw.model_ram_gb = detected_ram

    # Automatically synthesize system & hardware execution notes
    final_notes = generate_auto_system_notes(
        hw=hw,
        backend_name=backend,
        endpoint_url=endpoint_url,
        server_runtime=getattr(engine, "server_runtime", None),
        user_notes=notes,
    )
    if final_notes:
        logger.info(f"System & Hardware Notes: [italic]{final_notes}[/italic]")

    # 5. Parse Selected Tasks
    task_list = [t.strip() for t in tasks.split(",")] if tasks else None

    # 6. Run Benchmark Suite
    suite = BenchmarkSuite()
    result = suite.run_suite(
        backend=engine,
        model_meta=meta,
        hardware_info=hw,
        selected_tasks=task_list,
        sample_limit=sample_limit,
        notes=final_notes,
    )

    # 6. Save JSON Output
    out_path = Path(output_dir)
    if output_dir == "data/results":
        if Path("website/data/results").exists():
            out_path = Path("website/data/results")
        elif Path("Website/data/results").exists():
            out_path = Path("Website/data/results")
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"{meta.model_slug}.json"
    json_str = result.model_dump_json(indent=2)
    result_file.write_text(json_str, encoding="utf-8")
    logger.info(f"Standardized result successfully saved to: [bold cyan]{result_file}[/bold cyan]")

    # 7. Render Result Summary Table
    summary_table = Table(title=f"[Benchmark Summary: {meta.model_slug}]", show_header=True, header_style="bold cyan")
    summary_table.add_column("Category", style="bold")
    summary_table.add_column("Weight")
    summary_table.add_column("Normalized Score", style="bold yellow")
    for cat_name, cat in result.categories.items():
        summary_table.add_row(cat_name.title(), f"{cat.weight * 100:.0f}%", f"{cat.normalized_score:.2f}%")
    summary_table.add_row("[bold]Composite SLM-Index[/bold]", "100%", f"[bold green]{result.slm_index:.2f}%[/bold green]")
    console.print(summary_table)

    # 8. Automated Submission if requested
    if submit:
        if not github_repo:
            logger.error("--submit requires --github-repo (e.g., 'owner/leaderboard-repo')")
            raise typer.BadParameter("Missing --github-repo")
        
        logger.info(f"Preparing automated Pull Request submission to {github_repo}...")
        repo_url = f"https://github.com/{github_repo}.git" if not github_repo.startswith("http") else github_repo
        git_mgr = GitSubmissionManager(repo_url=repo_url, token=token, dry_run=dry_run)
        git_mgr.prepare_repo()
        branch_name = git_mgr.commit_and_push(
            model_slug=meta.model_slug,
            json_content=json_str,
            slm_index=result.slm_index,
        )

        pr_mgr = PullRequestManager(repo_identifier=github_repo, token=token, dry_run=dry_run)
        pr_info = pr_mgr.create_pull_request(branch_name=branch_name, result=result)
        console.print(Panel(
            f"Pull Request URL: {pr_info['pr_url']}\nHead Branch: {branch_name}\nTarget: {github_repo}:main",
            title="🚀 Submission Completed",
            border_style="green",
        ))

    return result


@app.command()
def batch(
    batch_file: Path = typer.Option(..., "--batch-file", "-f", help="Text file with model IDs (one per line) or JSON file"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Inference backend"),
    endpoint_url: str = typer.Option("http://localhost:11434/v1", "--endpoint-url", "--endpoint", "--base-url", help="OpenAI-compatible endpoint URL"),
    api_key: str = typer.Option("EMPTY", "--api-key", envvar="OPENAI_API_KEY", help="API key for endpoint"),
    model_ram_gb: Optional[float] = typer.Option(None, "--model-ram-gb", "--ram-gb", help="Size in RAM / VRAM in GB that the model takes up"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Optional notes on batch evaluation environment or runner"),
    quantization: str = typer.Option("BF16", "--quantization", "-q", help="Quantization method"),
    output_dir: str = typer.Option("data/results", "--output-dir", "-o", help="Target results directory"),
    sample_limit: Optional[int] = typer.Option(None, "--sample-limit", "-n", help="Sample limit per task"),
    submit: bool = typer.Option(False, "--submit", "-s", help="Submit PR for each model"),
    github_repo: Optional[str] = typer.Option(None, "--github-repo", help="Target GitHub repository"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN", help="GitHub Token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run"),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort", "--effort", help="Reasoning effort level for reasoning models"),
):
    """Run evaluation in batch mode for multiple models from a file."""
    setup_logger()
    if isinstance(reasoning_effort, typer.models.OptionInfo):
        reasoning_effort = None
    if not batch_file.exists():
        logger.error(f"Batch file not found: {batch_file}")
        raise typer.BadParameter(f"File not found: {batch_file}")

    lines = [line.strip() for line in batch_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    logger.info(f"Loaded {len(lines)} models from {batch_file} for batch evaluation.")

    if output_dir == "data/results":
        if Path("website/data/results").exists():
            output_dir = "website/data/results"
        elif Path("Website/data/results").exists():
            output_dir = "Website/data/results"

    results = []
    for idx, model_id in enumerate(lines, 1):
        logger.info(f"\n[bold green]Processing model [{idx}/{len(lines)}]: {model_id}[/bold green]")
        res = evaluate(
            model_id=model_id,
            quantization=quantization,
            backend=backend,
            endpoint_url=endpoint_url,
            api_key=api_key,
            model_ram_gb=model_ram_gb,
            output_dir=output_dir,
            sample_limit=sample_limit,
            tasks=None,
            submit=submit,
            github_repo=github_repo,
            token=token,
            dry_run=dry_run,
            allow_over_50b=True,
            notes=notes or "Batch evaluation",
            reasoning_effort=reasoning_effort,
        )
        results.append(res)

    logger.info(f"Batch run completed for {len(results)} models.")
    console.print(f"[bold green]Batch run completed for {len(results)} models.[/bold green]")


@app.command()
def submit(
    result_file: Path = typer.Option(..., "--result-file", "-f", help="Path to existing benchmark result JSON file"),
    github_repo: str = typer.Option(..., "--github-repo", "-r", help="Target GitHub repository (e.g., owner/repo)"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN", help="GitHub Personal Access Token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode"),
):
    """Submit an existing benchmark result JSON file to a GitHub repository via PR."""
    setup_logger()
    if not result_file.exists():
        logger.error(f"Result file not found: {result_file}")
        raise typer.BadParameter(f"Result file not found: {result_file}")

    data = json.loads(result_file.read_text(encoding="utf-8"))
    result = BenchmarkResult.model_validate(data)

    repo_url = f"https://github.com/{github_repo}.git" if not github_repo.startswith("http") else github_repo
    git_mgr = GitSubmissionManager(repo_url=repo_url, token=token, dry_run=dry_run)
    git_mgr.prepare_repo()
    branch_name = git_mgr.commit_and_push(
        model_slug=result.model.model_slug,
        json_content=result_file.read_text(encoding="utf-8"),
        slm_index=result.slm_index,
    )

    pr_mgr = PullRequestManager(repo_identifier=github_repo, token=token, dry_run=dry_run)
    pr_info = pr_mgr.create_pull_request(branch_name=branch_name, result=result)

    console.print(Panel(
        f"Pull Request URL: {pr_info['pr_url']}\nHead Branch: {branch_name}\nTarget: {github_repo}:main",
        title="🚀 Submission Completed",
        border_style="green",
    ))


@app.command()
def init_website(
    target_dir: Path = typer.Option(Path("./website"), "--target-dir", "-d", help="Directory where static website will be created"),
):
    """Generate or update static GitHub Pages leaderboard website template."""
    setup_logger()
    from .website_template import scaffold_website
    scaffold_website(target_dir)
    logger.info(f"Static website scaffolded successfully at [bold green]{target_dir}[/bold green]")


@app.command()
def tui(
    model_id: str = typer.Option("XHToken/Spark-X2.6-4B-INT4", "--model-id", "-m", help="Default model repository ID"),
    backend: str = typer.Option("openai", "--backend", "-b", help="Default inference backend"),
    endpoint_url: str = typer.Option("http://127.0.0.1:8000/v1", "--endpoint-url", "--endpoint", help="Server endpoint URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="API key"),
    quantization: str = typer.Option("INT4", "--quantization", "-q", help="Quantization method"),
    reasoning_effort: str = typer.Option("low", "--reasoning-effort", "--effort", help="Reasoning effort level"),
):
    """Launch the interactive Terminal User Interface (TUI) with real-time LLM output verification."""
    from .tui import launch_tui
    launch_tui(
        default_model=model_id,
        default_backend=backend,
        default_endpoint=endpoint_url,
        default_api_key=api_key,
        default_quant=quantization,
        default_effort=reasoning_effort,
    )


def main():
    if len(sys.argv) == 1:
        from .tui import launch_tui
        launch_tui()
    else:
        app()


if __name__ == "__main__":
    main()
