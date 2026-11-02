"""
Pull Request automation using PyGithub and GitHub REST API with formatted markdown summaries.
"""

import os
import re
import json
import requests
from typing import Optional, Dict, Any
from ..models.schema import BenchmarkResult
from ..utils.logger import logger


def generate_pr_markdown_body(result: BenchmarkResult) -> str:
    """Generate professional, comprehensive PR body for GitHub Pages leaderboard submissions."""
    m = result.model
    hw = result.hardware

    # Category rows
    cat_rows = []
    for cat_name, cat in result.categories.items():
        weight_pct = int(cat.weight * 100)
        cat_rows.append(f"| **{cat_name.title()}** ({weight_pct}% wt) | `{cat.normalized_score:.2f}%` | {cat.task_count} task(s) |")
    cat_table = "\n".join(cat_rows)

    # Task rows
    task_rows = []
    for tid, task in result.tasks.items():
        lat = f"{task.avg_latency_ms:.1f} ms" if task.avg_latency_ms else "N/A"
        tps = f"{task.avg_tokens_per_sec:.1f} t/s" if task.avg_tokens_per_sec else "N/A"
        task_rows.append(
            f"| `{task.task_id}` | {task.category.title()} | `{task.normalized_score:.2f}%` | "
            f"{task.passed_samples}/{task.total_samples} | {lat} | {tps} |"
        )
    task_table = "\n".join(task_rows)

    param_str = f"{m.param_count_b:.2f}B" if m.param_count_b is not None else "Unknown"
    vram_str = f"{hw.gpu_vram_gb:.1f} GB VRAM" if hw.gpu_vram_gb is not None else "N/A"

    body = f"""# 📊 Automated SLM Evaluation Submission: {m.model_id}

### 🌟 Aggregate SLM-Index: **`{result.slm_index:.2f}%`**

This pull request was automatically submitted by the **SLM-Bench Evaluation Harness**. It adds verified benchmark results to the static GitHub Pages leaderboard database.

---

## 🏷️ Model Specifications
| Attribute | Specification |
| :--- | :--- |
| **Model Repository** | [`{m.model_id}`](https://huggingface.co/{m.model_id}) |
| **Model Slug** | `{m.model_slug}` |
| **Parameter Count** | **{param_str}** ({m.param_tier}) |
| **Quantization** | `{m.quantization}` |
| **Architecture** | `{m.architecture or 'CausalLM'}` |
| **Context Length** | `{m.context_length} tokens` |
| **License** | `{m.license or 'unknown'}` |
| **Inference Backend** | `{m.backend}` |

---

## 📈 Benchmark Pillar Breakdown
| Category | Score | Task Details |
| :--- | :---: | :---: |
{cat_table}

---

## 🔬 Task-Level Performance
| Task | Category | Score | Passed / Total | Avg Latency | Tokens / Sec |
| :--- | :---: | :---: | :---: | :---: | :---: |
{task_table}

---

## 💻 Hardware & Runtime Environment
- **GPU**: {hw.gpu_name or 'N/A (CPU)'} ({vram_str})
- **CUDA Runtime**: {hw.cuda_version or 'N/A'} | **Driver**: {hw.driver_version or 'N/A'}
- **CPU**: {hw.cpu} ({hw.cpu_cores} logical cores)
- **System RAM**: {hw.total_ram_gb:.1f} GB
- **OS**: {hw.os}

---

## 🔄 Reproduction Command
```bash
slm-bench evaluate \\
  --model-id "{m.model_id}" \\
  --quantization "{m.quantization}" \\
  --backend "{m.backend}"
```
"""
    return body


def parse_repo_owner_name(repo_identifier: str) -> tuple[str, str]:
    """Parse 'owner/repo' or https://github.com/owner/repo(.git) into ('owner', 'repo')."""
    clean = repo_identifier.strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    
    if "github.com/" in clean:
        parts = clean.split("github.com/")[-1].split("/")
        return parts[0], parts[1]
    
    parts = clean.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    raise ValueError(f"Could not parse GitHub owner/repo from '{repo_identifier}'")


class PullRequestManager:
    """Manages opening Pull Requests via PyGithub or REST API."""

    def __init__(self, repo_identifier: str, token: Optional[str] = None, dry_run: bool = False):
        self.repo_identifier = repo_identifier
        self.token = token
        self.dry_run = dry_run
        self.owner, self.repo_name = parse_repo_owner_name(repo_identifier)

    def create_pull_request(
        self,
        branch_name: str,
        result: BenchmarkResult,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Open or simulate opening a Pull Request for benchmark submission."""
        title = f"[Benchmark] Submit {result.model.model_slug} (SLM-Index: {result.slm_index:.2f}%)"
        body = generate_pr_markdown_body(result)

        if self.dry_run:
            logger.info(f"[DRY-RUN] PR Title: {title}")
            logger.info(f"[DRY-RUN] PR Target: {self.owner}/{self.repo_name} (head: {branch_name} -> base: {base_branch})")
            logger.info("[DRY-RUN] PR Markdown Body preview:\n" + body[:400] + "\n...")
            return {
                "pr_url": f"https://github.com/{self.owner}/{self.repo_name}/pull/simulated-dry-run",
                "pr_number": 0,
                "title": title,
                "head": branch_name,
                "base": base_branch,
                "dry_run": True,
            }

        if not self.token:
            raise ValueError("A GitHub personal access token (GITHUB_TOKEN) is required to open a Pull Request.")

        # Try PyGithub first
        try:
            from github import Github
            gh = Github(self.token)
            gh_repo = gh.get_repo(f"{self.owner}/{self.repo_name}")
            pr = gh_repo.create_pull(
                title=title,
                body=body,
                head=branch_name,
                base=base_branch,
            )
            logger.info(f"Successfully opened Pull Request: [bold green]{pr.html_url}[/bold green]")
            return {
                "pr_url": pr.html_url,
                "pr_number": pr.number,
                "title": pr.title,
                "head": branch_name,
                "base": base_branch,
                "dry_run": False,
            }
        except Exception as e:
            logger.warning(f"PyGithub encounter exception ({e}). Falling back to direct GitHub REST API...")

        # Fallback to direct REST API
        api_url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/pulls"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "slm-bench/0.1.0",
        }
        payload = {
            "title": title,
            "body": body,
            "head": branch_name,
            "base": base_branch,
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            pr_data = resp.json()
            html_url = pr_data.get("html_url")
            logger.info(f"Successfully opened Pull Request via REST API: [bold green]{html_url}[/bold green]")
            return {
                "pr_url": html_url,
                "pr_number": pr_data.get("number"),
                "title": title,
                "head": branch_name,
                "base": base_branch,
                "dry_run": False,
            }
        else:
            raise RuntimeError(f"Failed to open PR on GitHub: {resp.status_code} - {resp.text}")
