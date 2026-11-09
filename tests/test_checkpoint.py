"""
Tests for replay checkpointing of inference calls.
"""

import json

from slm_bench.backends.base import BaseInferenceBackend, GenerationResult
from slm_bench.benchmarks.suite import BenchmarkSuite
from slm_bench.models.schema import HardwareInfo, ModelMetadata, QuestionResult, TaskMetric


class CountingBackend(BaseInferenceBackend):
    """Backend that counts real generation calls."""

    def __init__(self, model_id="test/model", quantization="GGUF_Q4_K_M"):
        super().__init__(model_id, quantization)
        self.calls = 0

    def load(self):
        self.is_loaded = True

    def generate(self, prompt, max_new_tokens=512, temperature=0.0, stop_sequences=None):
        self.calls += 1
        return GenerationResult(f"generated:{prompt}", 10.0, 4)

    def chat(self, messages, max_new_tokens=512, temperature=0.0, stop_sequences=None):
        self.calls += 1
        return GenerationResult(f"chat:{messages[-1]['content']}", 10.0, 4)


class FakeTask:
    task_id = "fake"
    category = "reasoning"

    def __init__(self, prompt="question one", max_new_tokens=8):
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def run(self, backend, sample_limit=None, on_sample=None, cancel_event=None, seed=42):
        gen = backend.chat(
            [{"role": "user", "content": self.prompt}],
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status="PARTIAL",
            raw_score=1.0,
            normalized_score=100.0,
            total_samples=1,
            passed_samples=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    raw_prompt=self.prompt,
                    raw_response=gen.text,
                    is_correct=True,
                )
            ],
        )


class GenerateFakeTask(FakeTask):
    def run(self, backend, sample_limit=None, on_sample=None, cancel_event=None, seed=42):
        gen = backend.generate(self.prompt, max_new_tokens=self.max_new_tokens, temperature=0.0)
        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status="PARTIAL",
            raw_score=1.0,
            normalized_score=100.0,
            total_samples=1,
            passed_samples=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    raw_prompt=self.prompt,
                    raw_response=gen.text,
                    is_correct=True,
                )
            ],
        )


def _meta(model_id="test/model"):
    return ModelMetadata(
        model_id=model_id,
        model_slug="model",
        param_count_b=1.0,
        param_tier="<3B",
        quantization="GGUF_Q4_K_M",
        backend="mock",
    )


def _run(tmp_path, backend, task=None, model_id="test/model"):
    suite = BenchmarkSuite()
    suite.resolve_tasks = lambda **kwargs: {"fake": task or FakeTask()}
    hw = HardwareInfo(os="Linux", cpu="Xeon", cpu_cores=8, total_ram_gb=16.0)
    return suite.run_suite(
        backend,
        _meta(model_id),
        hw,
        profile=None,
        checkpoint_path=str(tmp_path / "checkpoint.jsonl"),
        show_progress=False,
    )


def test_replays_cached_calls_without_touching_backend(tmp_path):
    cache = tmp_path / "checkpoint.jsonl"
    first_backend = CountingBackend()
    first = _run(tmp_path, first_backend)

    assert first_backend.calls == 1
    assert len(cache.read_text().splitlines()) == 1
    assert first.provenance["checkpoint"]["reused_calls"] == 0
    assert first.provenance["reused_samples"] == 0

    second_backend = CountingBackend()
    second = _run(tmp_path, second_backend)

    assert second_backend.calls == 0
    assert second.tasks["fake"].question_results[0].reused is True
    assert second.tasks["fake"].question_results[0].raw_response == "chat:question one"
    assert second.tasks["fake"].sub_metrics["reused_samples"] == 1
    assert second.provenance["reused_samples"] == 1
    assert second.provenance["checkpoint"]["reused_calls"] == 1
    assert second.provenance["checkpoint"]["loaded_entries"] == 1


def test_generate_calls_are_cached(tmp_path):
    _run(tmp_path, CountingBackend(), task=GenerateFakeTask())

    backend = CountingBackend()
    result = _run(tmp_path, backend, task=GenerateFakeTask())

    assert backend.calls == 0
    assert result.tasks["fake"].question_results[0].raw_response == "generated:question one"


def test_different_model_does_not_reuse(tmp_path):
    _run(tmp_path, CountingBackend(), model_id="test/model")

    backend = CountingBackend(model_id="other/model")
    _run(tmp_path, backend, model_id="other/model")

    assert backend.calls == 1


def test_different_generation_params_do_not_reuse(tmp_path):
    _run(tmp_path, CountingBackend(), task=FakeTask(max_new_tokens=8))

    backend = CountingBackend()
    _run(tmp_path, backend, task=FakeTask(max_new_tokens=16))

    assert backend.calls == 1


def test_malformed_checkpoint_lines_are_ignored(tmp_path):
    cache = tmp_path / "checkpoint.jsonl"
    cache.write_text('not json\n{"key": "broken"}\n')

    backend = CountingBackend()
    result = _run(tmp_path, backend)

    assert backend.calls == 1
    assert result.provenance["checkpoint"]["loaded_entries"] == 0


def test_checkpoint_appends_new_calls(tmp_path):
    _run(tmp_path, CountingBackend(), task=FakeTask(prompt="one"))

    backend = CountingBackend()
    result = _run(tmp_path, backend, task=FakeTask(prompt="two"))

    assert backend.calls == 1
    assert result.provenance["checkpoint"]["path"].endswith("checkpoint.jsonl")
    assert len((tmp_path / "checkpoint.jsonl").read_text().splitlines()) == 2
