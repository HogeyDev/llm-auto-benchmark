"""
Tests for Git automation, PR generation, and repository URL parsing.
"""

from slm_bench.submission.pr_manager import parse_repo_owner_name, generate_pr_markdown_body, PullRequestManager
from slm_bench.submission.git_manager import GitSubmissionManager
from slm_bench.models.schema import BenchmarkResult, ModelMetadata, HardwareInfo, TaskMetric, CategoryScore


def test_parse_repo_owner_name():
    assert parse_repo_owner_name("my-org/my-leaderboard") == ("my-org", "my-leaderboard")
    assert parse_repo_owner_name("https://github.com/my-org/my-leaderboard.git") == ("my-org", "my-leaderboard")
    assert parse_repo_owner_name("https://github.com/my-org/my-leaderboard") == ("my-org", "my-leaderboard")


def test_generate_pr_markdown_body():
    meta = ModelMetadata(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_slug="Qwen2.5-7B-Instruct-GGUF_Q4_K_M",
        param_count_b=7.62,
        param_tier="3B-8B",
        architecture="Qwen2ForCausalLM",
        quantization="GGUF_Q4_K_M",
        context_length=32768,
        license="apache-2.0",
        backend="llama.cpp",
    )
    hw = HardwareInfo(
        os="Windows 11",
        cpu="Intel i7",
        cpu_cores=16,
        total_ram_gb=32.0,
        gpu_name="NVIDIA RTX 5070",
        gpu_vram_gb=12.0,
        cuda_version="13.5",
    )
    task_m = TaskMetric(
        task_id="mmlu_pro",
        category="reasoning",
        status="FULL",
        raw_score=0.9,
        raw_metric_name="accuracy",
        normalized_score=90.0,
        total_samples=10,
        passed_samples=9,
        avg_latency_ms=120.0,
        avg_tokens_per_sec=45.0,
    )
    cat_s = CategoryScore(
        category="reasoning",
        weight=0.35,
        normalized_score=90.0,
        task_count=1,
    )
    result = BenchmarkResult(
        schema_version="2.0.0",
        evaluation_id="test-eval-id",
        status="FULL",
        eligible_for_leaderboard=True,
        model=meta,
        hardware=hw,
        tasks={"mmlu_pro": task_m},
        categories={"reasoning": cat_s},
        slm_index=90.0,
    )

    body = generate_pr_markdown_body(result)
    assert "Qwen/Qwen2.5-7B-Instruct" in body
    assert "90.00%" in body
    assert "GGUF_Q4_K_M" in body
    assert "NVIDIA RTX 5070" in body
    assert "slm-bench evaluate" in body


def test_git_submission_manager_dry_run(tmp_path):
    git_mgr = GitSubmissionManager(
        repo_url="https://github.com/test-org/test-leaderboard.git",
        token="fake-token",
        workspace_dir=str(tmp_path / "workspace"),
        dry_run=True,
    )
    path = git_mgr.prepare_repo()
    assert path is not None

    branch = git_mgr.commit_and_push(
        model_slug="test-slug",
        json_content="{}",
        slm_index=85.0,
    )
    assert branch == "eval/test-slug"


def test_pr_manager_dry_run():
    pr_mgr = PullRequestManager(repo_identifier="test-org/test-repo", dry_run=True)
    meta = ModelMetadata(
        model_id="test/model",
        model_slug="model-slug",
        param_tier="3B-8B",
        backend="mock",
    )
    hw = HardwareInfo(os="Linux", cpu="CPU", cpu_cores=4, total_ram_gb=8.0)
    result = BenchmarkResult(
        schema_version="1.0.0",
        evaluation_id="eval-1",
        model=meta,
        hardware=hw,
        tasks={},
        categories={},
        slm_index=75.0,
    )

    pr_info = pr_mgr.create_pull_request("eval/model-slug", result)
    assert pr_info["dry_run"] is True
    assert pr_info["pr_number"] == 0
    assert "simulated-dry-run" in pr_info["pr_url"]
