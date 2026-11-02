"""
Interactive Terminal User Interface (TUI) for SLM-Bench.
Provides an easy-to-use graphical console with:
- Automatic hardware and environment detection
- Curated model presets and customizable configuration controls
- Endpoint pre-flight health checking ("Test Connection")
- Smooth in-place live generation streaming (without fragmented line breaks)
- Structured, readable evaluation log with full AI responses and clean formatting
- Cooperative cancellation (Stop button terminates active benchmarks cleanly)
- Scoreboard results table with automatic clearing on re-runs
- Integrated Pull Request submission modal to GitHub
- Keyboard shortcuts and Help overlay modal
"""

import os
import sys
import json
import time
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, List

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    Label,
    Input,
    Select,
    Button,
    Checkbox,
    ProgressBar,
    RichLog,
    DataTable,
)
from textual.screen import ModalScreen
from textual import work

from .models.hardware import get_hardware_info, is_endpoint_wsl_relayed, is_wsl_environment, generate_auto_system_notes
from .models.metadata import fetch_hf_model_metadata
from .models.memory_detector import auto_detect_model_ram_gb
from .backends.registry import get_backend
from .benchmarks.suite import BenchmarkSuite
from .submission.git_manager import GitSubmissionManager
from .submission.pr_manager import PullRequestManager


MODEL_PRESETS = [
    ("Custom / Manual Entry", "custom"),
    ("Llama 3.2 1B Instruct (Meta)", "meta-llama/Llama-3.2-1B-Instruct"),
    ("Llama 3.2 3B Instruct (Meta)", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Qwen 2.5 1.5B Instruct (Alibaba)", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen 2.5 3B Instruct (Alibaba)", "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen 2.5 7B Instruct (Alibaba)", "Qwen/Qwen2.5-7B-Instruct"),
    ("Phi-3.5 Mini Instruct (Microsoft)", "microsoft/Phi-3.5-mini-instruct"),
    ("Gemma 2 2B IT (Google)", "google/gemma-2-2b-it"),
    ("Spark X2.6 4B INT4 (Local)", "XHToken/Spark-X2.6-4B-INT4"),
]


TUI_CSS = """
Screen {
    background: #0d1117;
    color: #e6edf3;
}

#layout {
    height: 100%;
}

#sidebar {
    width: 32%;
    min-width: 38;
    max-width: 48;
    height: 100%;
    background: #161b22;
    border-right: solid #30363d;
    padding: 1;
    overflow-y: scroll;
}

#main-content {
    height: 100%;
    padding: 1;
}

.box-title {
    color: #58a6ff;
    text-style: bold;
    margin-bottom: 1;
}

.section-label {
    color: #8b949e;
    margin-top: 1;
    text-style: bold;
}

.specs-card {
    background: #21262d;
    border: solid #30363d;
    padding: 1;
    margin-bottom: 1;
    color: #c9d1d9;
}

.status-card {
    background: #161b22;
    border: solid #30363d;
    padding: 1;
    margin-bottom: 1;
    height: auto;
}

#btn-run {
    width: 100%;
    margin-top: 1;
    background: #238636;
    color: white;
    text-style: bold;
}

#btn-run:hover {
    background: #2ea043;
}

#btn-stop {
    width: 100%;
    margin-top: 1;
    background: #da3633;
    color: white;
}

#btn-stop:hover {
    background: #f85149;
}

#btn-test-conn {
    width: 100%;
    margin-top: 1;
    margin-bottom: 1;
    background: #1f6feb;
    color: white;
}

#btn-test-conn:hover {
    background: #388bfd;
}

#btn-submit {
    width: 100%;
    margin-top: 1;
    background: #8957e5;
    color: white;
    text-style: bold;
}

#btn-submit:hover {
    background: #a371f7;
}

#btn-submit:disabled {
    background: #30363d;
    color: #8b949e;
}

#log-container {
    height: 1fr;
    border: solid #30363d;
    background: #0a0c10;
    margin-bottom: 1;
}

#live-stream-box {
    dock: top;
    height: auto;
    max-height: 8;
    background: #161b22;
    border-bottom: solid #30363d;
    padding: 1;
    color: #7ee787;
    overflow-y: scroll;
}

#stream-log {
    height: 1fr;
    background: #0a0c10;
    color: #e6edf3;
    overflow-y: scroll;
    padding: 0 1;
}

#progress-bar {
    margin-top: 1;
    margin-bottom: 1;
}

#results-table-container {
    height: 12;
    border: solid #30363d;
    background: #161b22;
}

.task-checkbox {
    margin-top: 0;
    margin-bottom: 0;
}

/* Modal styling */
.modal-container {
    width: 60;
    height: auto;
    background: #161b22;
    border: thick #58a6ff;
    padding: 2;
}

.modal-title {
    text-style: bold;
    color: #58a6ff;
    margin-bottom: 1;
    text-align: center;
}

.modal-actions {
    margin-top: 1;
    height: auto;
    align: right middle;
}

.modal-actions Button {
    margin-left: 1;
}
"""


class HelpModalScreen(ModalScreen):
    """Modal dialog displaying keyboard shortcuts and benchmark details."""

    def compose(self) -> ComposeResult:
        with Container(classes="modal-container"):
            yield Label("ℹ️ SLM-Bench Studio Help & Shortcuts", classes="modal-title")
            help_text = (
                "[bold cyan]Keyboard Shortcuts:[/bold cyan]\n"
                "  • [bold yellow]r[/bold yellow] : Start benchmark evaluation\n"
                "  • [bold yellow]s[/bold yellow] : Stop / cancel ongoing evaluation\n"
                "  • [bold yellow]b[/bold yellow] : Toggle left configuration sidebar\n"
                "  • [bold yellow]c[/bold yellow] : Clear generation stream log\n"
                "  • [bold yellow]?[/bold yellow] or [bold yellow]h[/bold yellow] : Show this help screen\n"
                "  • [bold yellow]q[/bold yellow] : Quit application\n\n"
                "[bold cyan]Scoring Weights (SLM-Index):[/bold cyan]\n"
                "  • Coding (Agentic): 30%\n"
                "  • Reasoning & Math: 25%\n"
                "  • Instruction Following: 25%\n"
                "  • Tool Calling: 20%\n\n"
                "[dim]Scores are saved to data/results/<slug>.json and are ready for PR submission.[/dim]"
            )
            yield Static(help_text)
            with Horizontal(classes="modal-actions"):
                yield Button("Close", variant="primary", id="help-btn-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-btn-close":
            self.dismiss(None)


class SubmitModalScreen(ModalScreen):
    """Modal dialog for submitting benchmark results via GitHub Pull Request."""

    def __init__(self, result_path: str, model_slug: str, slm_index: float, **kwargs):
        super().__init__(**kwargs)
        self.result_path = result_path
        self.model_slug = model_slug
        self.slm_index = slm_index

    def compose(self) -> ComposeResult:
        with Container(classes="modal-container"):
            yield Label(f"🚀 Submit Benchmark: {self.model_slug}", classes="modal-title")
            yield Label(f"Composite SLM-Index: [bold yellow]{self.slm_index:.2f}%[/bold yellow]\n")

            yield Label("Target GitHub Repository (owner/repo):", classes="section-label")
            yield Input(value="SeanDolan0/llm-auto-benchmark", id="modal-input-repo")

            yield Label("GitHub Personal Access Token (PAT):", classes="section-label")
            yield Input(value=os.environ.get("GITHUB_TOKEN", ""), placeholder="ghp_...", password=True, id="modal-input-token")

            yield Checkbox("Dry Run (Validate without pushing branch or PR)", value=False, id="modal-chk-dryrun")

            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", variant="default", id="modal-btn-cancel")
                yield Button("Submit PR", variant="success", id="modal-btn-submit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "modal-btn-cancel":
            self.dismiss(None)
        elif event.button.id == "modal-btn-submit":
            repo = self.query_one("#modal-input-repo", Input).value.strip()
            token = self.query_one("#modal-input-token", Input).value.strip() or None
            dry_run = self.query_one("#modal-chk-dryrun", Checkbox).value
            self.dismiss({
                "repo": repo,
                "token": token,
                "dry_run": dry_run,
                "result_path": self.result_path,
                "model_slug": self.model_slug,
                "slm_index": self.slm_index,
            })


class SLMBenchTUI(App):
    """Modern interactive TUI application for SLM-Bench."""

    TITLE = "SLM-Bench Studio"
    SUB_TITLE = "Automated Evaluation & Live Model Verification"
    CSS = TUI_CSS

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_eval", "Run"),
        ("s", "stop_eval", "Stop"),
        ("b", "toggle_sidebar", "Toggle Sidebar"),
        ("c", "clear_log", "Clear Log"),
        ("?", "show_help", "Help"),
        ("h", "show_help", "Help"),
    ]

    def __init__(
        self,
        default_model: str = "XHToken/Spark-X2.6-4B-INT4",
        default_backend: str = "openai",
        default_endpoint: str = "http://127.0.0.1:8000/v1",
        default_api_key: Optional[str] = None,
        default_effort: str = "low",
        default_quant: str = "INT4",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_model = default_model
        self.default_backend = default_backend
        self.default_endpoint = default_endpoint
        self.default_api_key = default_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.default_effort = default_effort
        self.default_quant = default_quant

        # Detected Hardware
        self.hw = get_hardware_info()
        self.eval_in_progress = False
        self.cancel_event = threading.Event()
        self.current_live_response = ""

        # Last completed evaluation result and persistence file
        self.last_result = None
        self.last_result_path = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="layout"):
            # Left Sidebar: Configuration Controls
            with ScrollableContainer(id="sidebar"):
                yield Label("⚡ Host System Hardware", classes="box-title")

                # Hardware Info Card
                cpu_short = self.hw.cpu.replace("Intel(R) Core(TM) ", "").replace("Processor", "").strip()
                gpu_short = (self.hw.gpu_name or "CPU Mode").replace("NVIDIA GeForce ", "")
                vram_str = f"{self.hw.gpu_vram_gb:.1f} GB VRAM" if self.hw.gpu_vram_gb else "No dedicated VRAM"
                os_str = "Windows 11 (WSL2)" if is_endpoint_wsl_relayed(self.default_endpoint) or is_wsl_environment() else self.hw.os.split("(")[0].strip()

                specs_text = (
                    f"[bold cyan]GPU:[/bold cyan] {gpu_short} ({vram_str})\n"
                    f"[bold cyan]CPU:[/bold cyan] {cpu_short}\n"
                    f"[bold cyan]RAM:[/bold cyan] {self.hw.total_ram_gb:.1f} GB | [bold cyan]OS:[/bold cyan] {os_str}"
                )
                yield Static(specs_text, classes="specs-card")

                yield Label("Model Presets:", classes="section-label")
                yield Select(
                    MODEL_PRESETS,
                    value=self.default_model if any(p[1] == self.default_model for p in MODEL_PRESETS) else "custom",
                    id="select-preset",
                )

                yield Label("Model ID / Hugging Face Repo:", classes="section-label")
                yield Input(value=self.default_model, placeholder="e.g. meta-llama/Llama-3.2-3B", id="input-model")

                yield Label("Quantization:", classes="section-label")
                yield Input(value=self.default_quant, placeholder="e.g. INT4, BF16, AWQ_4BIT", id="input-quant")

                yield Label("Inference Backend:", classes="section-label")
                yield Select(
                    [
                        ("OpenAI-Compatible (vLLM / Ollama)", "openai"),
                        ("Mock Engine (Test / Simulation)", "mock"),
                        ("HuggingFace Transformers", "transformers"),
                        ("Llama.cpp", "llamacpp"),
                    ],
                    value=self.default_backend,
                    id="select-backend",
                )

                yield Label("Server Endpoint URL:", classes="section-label")
                yield Input(value=self.default_endpoint, id="input-endpoint")
                yield Button("🔌 Test Connection", id="btn-test-conn")

                yield Label("API Key:", classes="section-label")
                yield Input(value=self.default_api_key, placeholder="(Optional for local backends)", password=True, id="input-apikey")

                yield Label("Reasoning Effort:", classes="section-label")
                yield Select(
                    [
                        ("low (Fast thinking)", "low"),
                        ("medium (Balanced)", "medium"),
                        ("high (Deep reasoning)", "high"),
                        ("none (Disabled)", "none"),
                    ],
                    value=self.default_effort,
                    id="select-effort",
                )

                yield Label("Sample Limit:", classes="section-label")
                yield Select(
                    [
                        ("1 sample (Quick test)", "1"),
                        ("3 samples (Fast check)", "3"),
                        ("5 samples (Standard)", "5"),
                        ("All samples (Full suite)", "all"),
                    ],
                    value="1",
                    id="select-limit",
                )

                yield Label("Evaluation Tasks:", classes="section-label")
                yield Checkbox("Reasoning & Math", value=True, id="chk-reasoning", classes="task-checkbox")
                yield Checkbox("Agentic Coding", value=True, id="chk-coding", classes="task-checkbox")
                yield Checkbox("Instruction Following", value=True, id="chk-instruction", classes="task-checkbox")
                yield Checkbox("Tool Calling", value=True, id="chk-tools", classes="task-checkbox")

                yield Button("▶  Start Evaluation", variant="success", id="btn-run")
                yield Button("⏹  Stop", variant="error", id="btn-stop", disabled=True)

            # Right Main Panel: Live Telemetry, Streaming Output, and Results
            with Vertical(id="main-content"):
                with Container(classes="status-card"):
                    yield Label("[bold cyan]Status:[/bold cyan] Ready to benchmark", id="lbl-status")
                    yield ProgressBar(total=100, show_eta=False, id="progress-bar")

                # Live Streaming Terminal & Clean Output Log
                yield Label("📺 Live LLM Generation & Verification Stream", classes="box-title")
                with Container(id="log-container"):
                    yield Static("[dim italic]Model output will stream live here during evaluation...[/dim italic]", id="live-stream-box")
                    yield RichLog(id="stream-log", wrap=True, highlight=False, markup=True)

                # Results Summary Table
                yield Label("📊 Benchmark Scoreboard", classes="box-title")
                with Container(id="results-table-container"):
                    table = DataTable(id="results-table", zebra_stripes=True)
                    table.add_columns("Category", "Task ID", "Score", "Accuracy / Pass@1", "Latency (ms)", "Tokens/s")
                    yield table

                yield Button("🚀  Submit Results to Leaderboard (PR)", variant="primary", id="btn-submit", disabled=True)

        yield Footer()

    def on_mount(self) -> None:
        """Initialize UI widgets upon mount."""
        log = self.query_one("#stream-log", RichLog)
        log.write("[bold green]Welcome to SLM-Bench Interactive Studio![/bold green]")
        log.write("[dim]Configure your model and server on the left, then click 'Start Evaluation' (or press 'r').[/dim]")
        log.write("[dim]Press '?' or 'h' for shortcuts and help.[/dim]")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update model input field when a preset is selected."""
        if event.select.id == "select-preset":
            val = str(event.value)
            if val != "custom":
                self.query_one("#input-model", Input).value = val

    def action_run_eval(self) -> None:
        """Trigger evaluation action."""
        if not self.eval_in_progress:
            self.start_evaluation()

    def action_stop_eval(self) -> None:
        """Trigger stop evaluation action."""
        if self.eval_in_progress:
            self.stop_evaluation()

    def action_toggle_sidebar(self) -> None:
        """Toggle left configuration sidebar visibility."""
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def action_clear_log(self) -> None:
        """Clear streaming output log and live preview."""
        log = self.query_one("#stream-log", RichLog)
        log.clear()
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[dim italic]Model output will stream live here during evaluation...[/dim italic]")

    def action_show_help(self) -> None:
        """Open the help and shortcuts modal."""
        self.push_screen(HelpModalScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn-run":
            self.start_evaluation()
        elif event.button.id == "btn-stop":
            self.stop_evaluation()
        elif event.button.id == "btn-test-conn":
            self.test_connection()
        elif event.button.id == "btn-submit":
            self.open_submit_modal()

    def test_connection(self) -> None:
        """Initiate endpoint connectivity pre-flight test."""
        endpoint_url = self.query_one("#input-endpoint", Input).value.strip() or self.default_endpoint
        api_key = self.query_one("#input-apikey", Input).value.strip()
        self.test_connection_worker(endpoint_url, api_key)

    @work(thread=True)
    def test_connection_worker(self, endpoint_url: str, api_key: str) -> None:
        """Pre-flight check to verify endpoint connectivity."""
        self.call_from_thread(self._update_status, "[bold yellow]Testing endpoint connection...[/bold yellow]")
        try:
            url = endpoint_url.rstrip("/")
            if not url.startswith("http"):
                url = f"http://{url}"

            test_url = f"{url}/models"
            req = urllib.request.Request(test_url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")

            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                latency_ms = (time.perf_counter() - t0) * 1000
                status_code = resp.status

            self.call_from_thread(
                self._update_status,
                f"[bold green]✓ Endpoint reachable[/bold green] (HTTP {status_code}, {latency_ms:.0f}ms)",
            )
            self.call_from_thread(
                self._log_info,
                f"[bold green]✓ Connection Test Passed:[/bold green] Reachable at {test_url} in {latency_ms:.1f}ms",
            )
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                msg = f"[bold yellow]⚠ Endpoint reachable (HTTP {e.code}: Authentication required)[/bold yellow]"
            elif e.code == 404:
                msg = "[bold green]✓ Server reachable[/bold green] (HTTP 404 on /models endpoint)"
            else:
                msg = f"[bold yellow]⚠ Server responded with HTTP {e.code}[/bold yellow]"
            self.call_from_thread(self._update_status, msg)
            self.call_from_thread(self._log_info, f"{msg} at {endpoint_url}")
        except Exception as e:
            err_msg = f"[bold red]✗ Connection failed: {e}[/bold red]"
            self.call_from_thread(self._update_status, err_msg)
            self.call_from_thread(
                self._log_info,
                f"[bold red]✗ Pre-flight check failed:[/bold red] Could not connect to {endpoint_url}. Verify that your local server (vLLM / Ollama) is active.",
            )

    def open_submit_modal(self) -> None:
        """Open the GitHub Pull Request submission modal."""
        if not self.last_result or not self.last_result_path:
            return

        def handle_submission(submission_data: Optional[Dict[str, Any]]) -> None:
            if submission_data:
                self.submit_pr_worker(submission_data)

        self.push_screen(
            SubmitModalScreen(
                result_path=self.last_result_path,
                model_slug=self.last_result.model.model_slug,
                slm_index=self.last_result.slm_index,
            ),
            handle_submission,
        )

    @work(thread=True)
    def submit_pr_worker(self, submission_data: Dict[str, Any]) -> None:
        """Execute Git commit and GitHub Pull Request creation in background."""
        repo = submission_data["repo"]
        token = submission_data.get("token")
        dry_run = submission_data.get("dry_run", False)
        res_path = Path(submission_data["result_path"])
        model_slug = submission_data["model_slug"]
        slm_index = submission_data["slm_index"]

        self.call_from_thread(self._update_status, "[bold yellow]Submitting results to GitHub...[/bold yellow]")
        try:
            json_content = res_path.read_text(encoding="utf-8")
            repo_url = f"https://github.com/{repo}.git" if not repo.startswith("http") else repo

            git_mgr = GitSubmissionManager(repo_url=repo_url, token=token, dry_run=dry_run)
            git_mgr.prepare_repo()
            branch_name = git_mgr.commit_and_push(
                model_slug=model_slug,
                json_content=json_content,
                slm_index=slm_index,
            )

            pr_url = None
            if not dry_run:
                pr_mgr = PullRequestManager(github_repo=repo, token=token)
                pr_url = pr_mgr.create_pr(
                    branch_name=branch_name,
                    model_slug=model_slug,
                    slm_index=slm_index,
                    is_update=git_mgr.is_update,
                )

            self.call_from_thread(self._on_submit_finished, pr_url, dry_run, branch_name)
        except Exception as e:
            self.call_from_thread(self._on_submit_error, str(e))

    def _on_submit_finished(self, pr_url: Optional[str], dry_run: bool, branch_name: str) -> None:
        """Handle successful PR submission."""
        lbl = self.query_one("#lbl-status", Label)
        log = self.query_one("#stream-log", RichLog)
        if dry_run:
            lbl.update(f"[bold green]✓ Dry-run completed![/bold green] Branch: [cyan]{branch_name}[/cyan]")
            log.write(f"[bold green]✓ Dry-run submission completed successfully![/bold green]")
            log.write(f"Validated JSON schema and git branch: [bold cyan]{branch_name}[/bold cyan]")
        else:
            lbl.update(f"[bold green]✓ Pull Request Created![/bold green] {pr_url}")
            log.write(f"[bold green]🚀 PULL REQUEST OPENED SUCCESSFULLY![/bold green]")
            log.write(f"Leaderboard PR URL: [bold cyan]{pr_url}[/bold cyan]")

    def _on_submit_error(self, err_msg: str) -> None:
        """Handle submission failure."""
        lbl = self.query_one("#lbl-status", Label)
        lbl.update(f"[bold red]Submission failed: {err_msg}[/bold red]")
        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold red]✗ PR Submission Error:[/bold red] {err_msg}")

    def start_evaluation(self) -> None:
        """Start benchmark run in background worker thread."""
        if self.eval_in_progress:
            return

        self.eval_in_progress = True
        self.cancel_event.clear()
        self.current_live_response = ""

        self.query_one("#btn-run", Button).disabled = True
        self.query_one("#btn-stop", Button).disabled = False
        self.query_one("#btn-submit", Button).disabled = True

        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[bold yellow]⚡ Initializing model and inference backend...[/bold yellow]")

        # Clear scoreboard for a fresh run
        table = self.query_one("#results-table", DataTable)
        table.clear()

        status_lbl = self.query_one("#lbl-status", Label)
        status_lbl.update("[bold yellow]Initializing benchmark backend and model metadata...[/bold yellow]")

        # Read form configuration
        model_id = self.query_one("#input-model", Input).value.strip() or self.default_model
        quantization = self.query_one("#input-quant", Input).value.strip() or self.default_quant
        backend_name = str(self.query_one("#select-backend", Select).value)
        endpoint_url = self.query_one("#input-endpoint", Input).value.strip() or self.default_endpoint
        api_key = self.query_one("#input-apikey", Input).value.strip() or self.default_api_key
        effort_val = str(self.query_one("#select-effort", Select).value)
        effort = None if effort_val == "none" else effort_val

        limit_val = str(self.query_one("#select-limit", Select).value)
        sample_limit = None if limit_val == "all" else int(limit_val)

        selected_tasks = []
        if self.query_one("#chk-reasoning", Checkbox).value:
            selected_tasks.append("reasoning")
        if self.query_one("#chk-coding", Checkbox).value:
            selected_tasks.append("coding")
        if self.query_one("#chk-instruction", Checkbox).value:
            selected_tasks.append("instruction")
        if self.query_one("#chk-tools", Checkbox).value:
            selected_tasks.append("tool_calling")

        if not selected_tasks:
            selected_tasks = ["reasoning"]

        # Run worker
        self.run_eval_worker(
            model_id=model_id,
            quantization=quantization,
            backend_name=backend_name,
            endpoint_url=endpoint_url,
            api_key=api_key,
            reasoning_effort=effort,
            sample_limit=sample_limit,
            selected_tasks=selected_tasks,
        )

    def stop_evaluation(self) -> None:
        """Signal cooperative cancellation to running benchmark worker."""
        if not self.eval_in_progress:
            return

        self.cancel_event.set()
        self.query_one("#btn-stop", Button).disabled = True
        self.query_one("#lbl-status", Label).update("[bold red]Stopping evaluation...[/bold red]")
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[bold red]⏹ Cancellation requested. Waiting for active operation to stop...[/bold red]")
        log = self.query_one("#stream-log", RichLog)
        log.write("[bold yellow]Cancellation signal sent. Halting tasks...[/bold yellow]")

    @work(thread=True)
    def run_eval_worker(
        self,
        model_id: str,
        quantization: str,
        backend_name: str,
        endpoint_url: str,
        api_key: str,
        reasoning_effort: Optional[str],
        sample_limit: Optional[int],
        selected_tasks: List[str],
    ) -> None:
        """Background thread worker executing evaluation suite with cooperative cancellation."""
        try:
            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled before start.")

            # 1. Fetch metadata
            meta = fetch_hf_model_metadata(model_id, quantization=quantization, backend=backend_name)
            if reasoning_effort:
                meta.reasoning_effort = reasoning_effort

            hw = get_hardware_info(model_ram_gb=meta.model_ram_gb)

            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled during hardware detection.")

            # 2. Instantiate backend
            backend = get_backend(
                backend_name,
                model_id=model_id,
                quantization=quantization,
                endpoint_url=endpoint_url,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
            )

            # Hook token stream callback to verify active LLM generation
            def on_token_chunk(token: str):
                self.call_from_thread(self._append_token_chunk, token)

            backend.set_stream_callback(on_token_chunk)
            backend.load()

            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled after backend loading.")

            # Auto-detect RAM
            detected_ram = auto_detect_model_ram_gb(
                backend_instance=backend,
                backend_name=backend_name,
                model_id=model_id,
                quantization=quantization,
                param_count_b=meta.param_count_b,
                endpoint_url=endpoint_url if backend_name == "openai" else None,
            )
            if detected_ram:
                meta.model_ram_gb = detected_ram
                hw.model_ram_gb = detected_ram

            # Generate auto notes
            notes = generate_auto_system_notes(
                hw=hw,
                backend_name=backend_name,
                endpoint_url=endpoint_url,
                server_runtime=getattr(backend, "server_runtime", None),
            )

            # 3. Setup hooks for suite execution
            total_tasks_count = len(selected_tasks)
            current_task_idx = [0]

            def on_task_start(task_id: str, category: str):
                current_task_idx[0] += 1
                msg = f"Task [{current_task_idx[0]}/{total_tasks_count}]: [bold cyan]{task_id}[/bold cyan] ({category.title()})"
                pct = int((current_task_idx[0] - 1) / total_tasks_count * 100)
                self.call_from_thread(self._update_task_progress, msg, pct)
                self.call_from_thread(self._log_task_header, task_id, category)

            def on_sample(info: Dict[str, Any]):
                self.call_from_thread(self._log_sample_result, info)

            def on_task_complete(task_id: str, metric: Any):
                self.call_from_thread(self._add_metric_row, task_id, metric)

            suite = BenchmarkSuite()
            result = suite.run_suite(
                backend=backend,
                model_meta=meta,
                hardware_info=hw,
                selected_tasks=selected_tasks,
                sample_limit=sample_limit,
                notes=notes,
                on_task_start=on_task_start,
                on_sample=on_sample,
                on_task_complete=on_task_complete,
                show_progress=False,
                cancel_event=self.cancel_event,
            )

            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled during benchmark execution.")

            # 4. Save result
            out_dir = Path("data/results")
            out_dir.mkdir(parents=True, exist_ok=True)
            res_file = out_dir / f"{meta.model_slug}.json"
            res_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

            # Final UI notification
            self.call_from_thread(self._on_eval_finished, result, str(res_file))

        except InterruptedError:
            self.call_from_thread(self._on_eval_cancelled)
        except Exception as e:
            if self.cancel_event.is_set():
                self.call_from_thread(self._on_eval_cancelled)
            else:
                self.call_from_thread(self._on_eval_error, str(e))

    def _update_task_progress(self, msg: str, pct: int) -> None:
        """Update progress bar and status label from worker thread."""
        lbl = self.query_one("#lbl-status", Label)
        lbl.update(f"[bold green]Running:[/bold green] {msg}")
        pbar = self.query_one("#progress-bar", ProgressBar)
        pbar.progress = pct

    def _append_token_chunk(self, token: str) -> None:
        """Update live generation view in-place without artificial line breaks."""
        self.current_live_response += token
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update(f"[bold cyan]⚡ Live Generation Stream:[/bold cyan]\n{self.current_live_response}▌")

    def _log_task_header(self, task_id: str, category: str) -> None:
        """Log prominent task header banner."""
        self.current_live_response = ""
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update(f"[bold cyan]⚡ Evaluating Task:[/bold cyan] [bold yellow]{task_id}[/bold yellow] ([italic]{category.title()}[/italic])...")

        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold magenta]════════════════════════════════════════════════════════════════════════════[/bold magenta]")
        log.write(f"[bold cyan]▶ STARTING TASK:[/bold cyan] [bold yellow]{task_id}[/bold yellow] ([italic]{category.title()}[/italic])")
        log.write(f"[bold magenta]────────────────────────────────────────────────────────────────────────────[/bold magenta]")

    def _log_sample_result(self, info: Dict[str, Any]) -> None:
        """Log structured sample evaluation and clean AI output."""
        self.current_live_response = ""
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[dim italic]Model output will stream live here during evaluation...[/dim italic]")

        log = self.query_one("#stream-log", RichLog)
        idx = info.get("sample_index", 1)
        tot = info.get("total_samples", 1)
        passed = info.get("passed", False)
        lat = info.get("latency_ms", 0.0)
        toks = info.get("tokens", 0)
        tps = (toks / (lat / 1000.0)) if lat > 0 and toks > 0 else 0.0
        task_id = info.get("task_id", "")
        category = info.get("category", "")
        prompt = info.get("prompt", "")
        response = info.get("response", "")
        expected = info.get("expected", "")
        extracted = info.get("extracted", "")

        verdict_tag = "[bold green]✓ PASSED[/bold green]" if passed else "[bold red]✗ FAILED[/bold red]"

        log.write(f"[bold cyan]▶ [{category.title()}] {task_id}[/bold cyan] [dim](Sample {idx}/{tot})[/dim] {verdict_tag} [dim]| {lat:.0f}ms | {tps:.1f} tok/s[/dim]")
        if prompt:
            p_clean = prompt.strip().replace("\r\n", "\n")
            log.write(f"[dim]Prompt:[/dim] {p_clean}")
        if response:
            log.write("[bold green]AI Output:[/bold green]")
            for line in response.strip().splitlines():
                cleaned = line.rstrip()
                if cleaned:
                    log.write(f"  {cleaned}")
                else:
                    log.write("")
        if expected is not None or extracted is not None:
            log.write(f"[dim]Expected: {expected} | Extracted: {extracted}[/dim]")
        log.write("[dim]────────────────────────────────────────────────────────────────────────────[/dim]")

    def _add_metric_row(self, task_id: str, metric: Any) -> None:
        """Add finished metric to the results table."""
        table = self.query_one("#results-table", DataTable)
        score_styled = f"[bold green]{metric.normalized_score:.1f}%[/bold green]" if metric.normalized_score >= 70 else f"[bold yellow]{metric.normalized_score:.1f}%[/bold yellow]"
        table.add_row(
            metric.category.title(),
            task_id,
            score_styled,
            f"{metric.passed_samples}/{metric.total_samples}",
            f"{metric.avg_latency_ms:.1f}" if metric.avg_latency_ms else "N/A",
            f"{metric.avg_tokens_per_sec:.1f}" if metric.avg_tokens_per_sec else "N/A",
        )

    def _update_status(self, text: str) -> None:
        """Update top status message."""
        lbl = self.query_one("#lbl-status", Label)
        lbl.update(text)

    def _log_info(self, text: str) -> None:
        """Write an informational message to the stream log."""
        log = self.query_one("#stream-log", RichLog)
        log.write(text)

    def _on_eval_finished(self, result: Any, file_path: str) -> None:
        """Update UI upon evaluation completion."""
        self.eval_in_progress = False
        self.last_result = result
        self.last_result_path = file_path

        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        self.query_one("#btn-submit", Button).disabled = False

        pbar = self.query_one("#progress-bar", ProgressBar)
        pbar.progress = 100

        lbl = self.query_one("#lbl-status", Label)
        lbl.update(f"[bold green]✓ Complete![/bold green] Composite SLM-Index: [bold yellow]{result.slm_index:.2f}%[/bold yellow] | Saved: [cyan]{file_path}[/cyan]")

        live_box = self.query_one("#live-stream-box", Static)
        live_box.update(f"[bold green]✓ Evaluation completed![/bold green] Composite SLM-Index: [bold yellow]{result.slm_index:.2f}%[/bold yellow]")

        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold green]════════════════════════════════════════════════════════════════════════════[/bold green]")
        log.write(f"[bold green]✓ BENCHMARK COMPLETED SUCCESSFULLY![/bold green]")
        log.write(f"Aggregate SLM-Index Score: [bold yellow]{result.slm_index:.2f}%[/bold yellow]")
        log.write(f"Standardized Result JSON: [bold cyan]{file_path}[/bold cyan]")
        log.write(f"[bold green]════════════════════════════════════════════════════════════════════════════[/bold green]")

    def _on_eval_cancelled(self) -> None:
        """Handle clean cooperative cancellation."""
        self.eval_in_progress = False
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True

        lbl = self.query_one("#lbl-status", Label)
        lbl.update("[bold yellow]Evaluation cancelled by user.[/bold yellow]")

        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[bold yellow]⚠ Evaluation was cancelled by user.[/bold yellow]")

        log = self.query_one("#stream-log", RichLog)
        log.write("[bold yellow]⚠ Evaluation was cancelled by user. Background tasks stopped cleanly.[/bold yellow]")

    def _on_eval_error(self, err_msg: str) -> None:
        """Handle evaluation error."""
        self.eval_in_progress = False
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True

        lbl = self.query_one("#lbl-status", Label)
        lbl.update(f"[bold red]Error: {err_msg}[/bold red]")

        live_box = self.query_one("#live-stream-box", Static)
        live_box.update(f"[bold red]✖ Evaluation error: {err_msg}[/bold red]")

        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold red]✖ Evaluation Error Encountered:[/bold red] {err_msg}")


def launch_tui(**kwargs) -> None:
    """Launch the interactive SLM-Bench TUI application."""
    app = SLMBenchTUI(**kwargs)
    app.run()
