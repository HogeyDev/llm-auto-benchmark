"""
Rich logging and console utilities for SLM-Bench.
"""

import sys
import logging
import warnings
from rich.console import Console
from rich.logging import RichHandler

# Suppress harmless requests version mismatch warnings
try:
    import urllib3
    warnings.filterwarnings("ignore", message=".*urllib3.*")
except Exception:
    pass

# Ensure UTF-8 output on Windows consoles to prevent charmap UnicodeEncodeErrors
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure structured rich logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=True, show_path=False)],
    )
    logger = logging.getLogger("slm_bench")
    logger.setLevel(level)
    return logger


logger = logging.getLogger("slm_bench")
