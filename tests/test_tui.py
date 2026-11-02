"""
Tests for Textual Terminal User Interface (TUI).
"""

import os
import pytest
from textual.widgets import Select, Button, Input, DataTable
from slm_bench.tui import SLMBenchTUI, HelpModalScreen, SubmitModalScreen


@pytest.mark.anyio
async def test_tui_initialization_and_mounting():
    """Verify that the TUI app mounts widgets and exposes all configuration controls."""
    # Ensure no leaked hardcoded default key
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    app = SLMBenchTUI(
        default_model="meta-llama/Llama-3.2-3B-Instruct",
        default_quant="BF16",
        default_effort="medium",
    )
    async with app.run_test() as pilot:
        assert app.eval_in_progress is False
        assert "sk_live_" not in app.default_api_key
        assert app.query_one("#input-model", Input).value == "meta-llama/Llama-3.2-3B-Instruct"
        assert app.query_one("#input-quant", Input).value == "BF16"
        assert app.query_one("#btn-run", Button) is not None
        assert app.query_one("#btn-stop", Button).disabled is True
        assert app.query_one("#btn-test-conn", Button) is not None
        assert app.query_one("#btn-submit", Button).disabled is True
        assert app.query_one("#live-stream-box") is not None
        assert app.query_one("#stream-log") is not None
        assert app.query_one("#results-table", DataTable) is not None


@pytest.mark.anyio
async def test_tui_preset_selection():
    """Verify selecting a model preset updates the model ID input."""
    app = SLMBenchTUI()
    async with app.run_test() as pilot:
        select = app.query_one("#select-preset", Select)
        select.value = "Qwen/Qwen2.5-1.5B-Instruct"
        await pilot.pause()
        assert app.query_one("#input-model", Input).value == "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.anyio
async def test_tui_cancellation_logic():
    """Verify stop_evaluation triggers cancellation event and UI update."""
    app = SLMBenchTUI()
    async with app.run_test() as pilot:
        assert not app.cancel_event.is_set()
        app.eval_in_progress = True
        app.stop_evaluation()
        assert app.cancel_event.is_set()
        assert app.query_one("#btn-stop", Button).disabled is True


@pytest.mark.anyio
async def test_tui_table_clearing_on_start():
    """Verify results table is cleared when starting a new evaluation."""
    app = SLMBenchTUI(default_backend="mock")
    async with app.run_test() as pilot:
        table = app.query_one("#results-table", DataTable)
        table.add_row("Reasoning", "slm_reasoning_cot", "100%", "5/5", "10ms", "50")
        assert table.row_count == 1

        # Calling start_evaluation clears table
        # We set cancel_event to stop the worker immediately
        app.cancel_event.set()
        app.start_evaluation()
        assert table.row_count == 0


@pytest.mark.anyio
async def test_tui_streaming_and_sample_logging():
    """Verify in-place streaming updates live-stream-box and sample results are logged cleanly."""
    from textual.widgets import Static, RichLog
    app = SLMBenchTUI()
    async with app.run_test() as pilot:
        app._append_token_chunk("Hello ")
        app._append_token_chunk("world!")
        live_box = app.query_one("#live-stream-box", Static)
        assert "Hello world!" in str(live_box.render())

        app._log_sample_result({
            "task_id": "test_task",
            "category": "reasoning",
            "sample_index": 1,
            "total_samples": 1,
            "passed": True,
            "latency_ms": 20.0,
            "tokens": 5,
            "prompt": "Say hello",
            "response": "Hello world!",
            "expected": "Hello world!",
            "extracted": "Hello world!",
        })
        log = app.query_one("#stream-log", RichLog)
        assert len(log.lines) > 0

