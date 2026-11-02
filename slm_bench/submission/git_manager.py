"""
Git repository automation for benchmark submission branches and commits.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from ..utils.logger import logger


class GitSubmissionManager:
    """Manages Git branch creation, result file saving, committing, and pushing."""

    def __init__(
        self,
        repo_url: str,
        token: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.repo_url = repo_url
        self.token = token
        self.dry_run = dry_run
        self.workspace_dir = Path(workspace_dir or "./.slm_bench_repo").resolve()

    def _get_authenticated_url(self) -> str:
        """Inject GitHub token into remote URL if provided."""
        if not self.token or "@" in self.repo_url:
            return self.repo_url

        url = self.repo_url.strip()
        if url.startswith("https://"):
            return f"https://x-access-token:{self.token}@{url[8:]}"
        return url

    def _run_git(self, args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a git CLI command with error logging."""
        run_cwd = cwd or self.workspace_dir
        cmd_str = "git " + " ".join(args)
        logger.debug(f"Running git command: {cmd_str} in {run_cwd}")

        proc = subprocess.run(
            ["git"] + args,
            cwd=str(run_cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.error(f"Git command failed: {proc.stderr.strip()}")
            raise RuntimeError(f"Git failed: {proc.stderr.strip()}")
        return proc

    def prepare_repo(self, base_branch: str = "main") -> Path:
        """Clone or pull the target repository."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Simulating repository preparation for: {self.repo_url}")
            return self.workspace_dir

        auth_url = self._get_authenticated_url()

        if not (self.workspace_dir / ".git").exists():
            logger.info(f"Cloning {self.repo_url} into {self.workspace_dir}...")
            if self.workspace_dir.exists():
                shutil.rmtree(self.workspace_dir)
            self.workspace_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run_git(["clone", auth_url, str(self.workspace_dir)], cwd=self.workspace_dir.parent)
        else:
            logger.info(f"Pulling latest changes in {self.workspace_dir}...")
            self._run_git(["checkout", base_branch])
            self._run_git(["pull", "origin", base_branch])

        return self.workspace_dir

    def commit_and_push(
        self,
        model_slug: str,
        json_content: str,
        slm_index: float,
        branch_name: Optional[str] = None,
        base_branch: str = "main",
    ) -> str:
        """
        Create isolated branch, write data/results/<model_slug>.json, commit and push.
        Returns the branch name.
        """
        target_branch = branch_name or f"eval/{model_slug}"
        commit_message = f"Add benchmark result: {model_slug} (SLM-Index: {slm_index:.2f}%)"

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would create branch: {target_branch}")
            logger.info(f"[DRY-RUN] Would write JSON data for {model_slug} to data/results/{model_slug}.json")
            logger.info(f"[DRY-RUN] Would commit with message: '{commit_message}'")
            logger.info(f"[DRY-RUN] Would push branch {target_branch} to origin")
            return target_branch

        # Checkout fresh branch from base
        self._run_git(["checkout", base_branch])
        # Delete local branch if already exists
        try:
            self._run_git(["branch", "-D", target_branch])
        except Exception:
            pass
        self._run_git(["checkout", "-b", target_branch])

        # Write result file - prefer website/data/results or Website/data/results if present
        if (self.workspace_dir / "website" / "data" / "results").exists() or (self.workspace_dir / "website").exists():
            results_dir = self.workspace_dir / "website" / "data" / "results"
        elif (self.workspace_dir / "Website" / "data" / "results").exists() or (self.workspace_dir / "Website").exists():
            results_dir = self.workspace_dir / "Website" / "data" / "results"
        else:
            results_dir = self.workspace_dir / "data" / "results"

        results_dir.mkdir(parents=True, exist_ok=True)
        target_file = results_dir / f"{model_slug}.json"
        target_file.write_text(json_content, encoding="utf-8")
        logger.info(f"Saved benchmark result to {target_file}")

        # Git add, commit, push
        self._run_git(["add", str(target_file)])
        self._run_git(["config", "user.name", "SLM-Bench Bot"])
        self._run_git(["config", "user.email", "slm-bench-bot@users.noreply.github.com"])
        self._run_git(["commit", "-m", commit_message])

        auth_url = self._get_authenticated_url()
        self._run_git(["push", "-u", auth_url, target_branch, "--force"])
        logger.info(f"Pushed branch {target_branch} to origin.")

        return target_branch
