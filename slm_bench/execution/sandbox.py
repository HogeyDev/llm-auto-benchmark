"""
Secure Sandboxed Execution Environment for Model-Generated Code.
Enforces isolated container-based execution with:
- No network access (--network none)
- CPU limits (--cpus 1.0)
- Memory limits (--memory 512m)
- Wall-clock timeouts
- Ephemeral temporary directories
- No access to host credentials or docker socket

If container isolation is unavailable, marks evaluations as unavailable
rather than executing untrusted code on the host process.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from ..utils.logger import logger


class ContainerSandbox:
    """Isolated Docker container sandbox for executing untrusted model code."""

    def __init__(self, image: str = "python:3.11-slim", timeout_sec: int = 5, memory_limit: str = "512m", cpu_limit: str = "1.0"):
        self.image = image
        self.timeout_sec = timeout_sec
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._docker_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Docker is installed and the Docker daemon is responding."""
        if self._docker_available is not None:
            return self._docker_available
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                text=True,
            )
            self._docker_available = (res.returncode == 0)
        except Exception:
            self._docker_available = False
        return self._docker_available

    def run_code(
        self,
        code: str,
        test_script: str,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str, float]:
        """
        Execute generated code combined with test_script inside an isolated container.
        Returns:
            (passed: bool, output_or_error: str, execution_time_sec: float)
        """
        if not self.is_available():
            return False, "Docker container sandbox is unavailable. Host execution of untrusted code is blocked for security.", 0.0

        exec_timeout = timeout or self.timeout_sec
        temp_dir = tempfile.mkdtemp(prefix="slm_sandbox_")

        try:
            # Combine code and test runner
            full_script = (
                "# Model generated code\n"
                f"{code}\n\n"
                "# Official hidden test validation\n"
                f"{test_script}\n"
            )
            script_path = Path(temp_dir) / "run_eval.py"
            script_path.write_text(full_script, encoding="utf-8")

            # Run docker run with tight isolation
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", self.memory_limit,
                "--cpus", self.cpu_limit,
                "--pids-limit", "64",
                "-v", f"{os.path.abspath(temp_dir)}:/sandbox:ro",
                "-w", "/sandbox",
                self.image,
                "python", "-u", "/sandbox/run_eval.py",
            ]

            import time
            start = time.perf_counter()
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=exec_timeout,
                text=True,
            )
            elapsed = time.perf_counter() - start

            if proc.returncode == 0:
                return True, proc.stdout.strip(), elapsed
            else:
                err_msg = proc.stderr.strip() or proc.stdout.strip()
                return False, f"Execution failed (code {proc.returncode}): {err_msg}", elapsed

        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {exec_timeout}s.", float(exec_timeout)
        except Exception as e:
            return False, f"Sandbox execution error: {e}", 0.0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
