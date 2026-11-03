"""
Central Benchmark Registry and Official Evaluation Profiles.
Provides discovery and instantiation for established official benchmarks.
All custom handcrafted evaluations have been completely removed.
"""

from typing import Dict, List, Optional, Type, Callable, Any
from .base import BaseBenchmarkTask

# Official benchmark adapters
from .reasoning.mmlu_pro import MMLUProBenchmark
from .reasoning.gpqa import GPQADiamondBenchmark
from .reasoning.bbh import BBHBenchmark
from .mathematics.gsm8k import GSM8KBenchmark
from .mathematics.math500 import MATH500Benchmark
from .instruction.ifeval import IFEvalBenchmark
from .coding.evalplus import EvalPlusBenchmark
from .coding.livecodebench import LiveCodeBenchBenchmark
from .tools.bfcl import BFCLBenchmark
from .context.ruler import RULERBenchmark
from .safety.safety_adapter import SafetyBenchmark


# Registry mapping canonical task IDs to Task classes or factory callables
BENCHMARK_REGISTRY: Dict[str, Any] = {
    # Priority 1: Reasoning & Knowledge
    "mmlu_pro": MMLUProBenchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "bbh": BBHBenchmark,

    # Priority 2: Mathematics
    "gsm8k": GSM8KBenchmark,
    "math500": MATH500Benchmark,

    # Priority 3: Coding
    "evalplus": lambda: EvalPlusBenchmark(subset="humanevalplus"),
    "evalplus_humanevalplus": lambda: EvalPlusBenchmark(subset="humanevalplus"),
    "evalplus_mbppplus": lambda: EvalPlusBenchmark(subset="mbppplus"),
    "livecodebench": LiveCodeBenchBenchmark,

    # Priority 4: Instruction Following
    "ifeval": IFEvalBenchmark,

    # Priority 5: Function Calling
    "bfcl": BFCLBenchmark,

    # Priority 6: Long-Context
    "ruler": RULERBenchmark,

    # Priority 7: Safety
    "xstest": SafetyBenchmark,
}

# User convenience aliases mapped to canonical official tasks
TASK_ALIASES: Dict[str, str] = {
    "gpqa": "gpqa_diamond",
    "math": "math500",
    "instruction": "ifeval",
    "coding": "evalplus",
    "code": "evalplus",
    "humaneval": "evalplus",
    "tools": "bfcl",
    "tool_calling": "bfcl",
    "safety": "xstest",
    "context": "ruler",
}

# Evaluation Profiles
EVALUATION_PROFILES: Dict[str, List[str]] = {
    # Quick: development & smoke testing only. Uses seeded samples, marked PARTIAL.
    "quick": [
        "mmlu_pro",
        "gpqa_diamond",
        "gsm8k",
        "math500",
        "ifeval",
    ],
    # Standard: complete official datasets for core reasoning, math, and instruction benchmarks.
    "standard": [
        "mmlu_pro",
        "gpqa_diamond",
        "bbh",
        "gsm8k",
        "math500",
        "ifeval",
    ],
    # Extended: complete official suite including coding, tools, long-context, and safety.
    "extended": [
        "mmlu_pro",
        "gpqa_diamond",
        "bbh",
        "gsm8k",
        "math500",
        "ifeval",
        "evalplus",
        "bfcl",
        "ruler",
        "xstest",
    ],
}


def get_task(task_id: str) -> BaseBenchmarkTask:
    """Instantiate and return benchmark task instance by ID or alias."""
    clean_id = task_id.lower().strip()
    canonical_id = TASK_ALIASES.get(clean_id, clean_id)

    if canonical_id not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unrecognized benchmark task '{task_id}'. "
            f"Available official benchmarks: {list(BENCHMARK_REGISTRY.keys())}"
        )

    task_factory = BENCHMARK_REGISTRY[canonical_id]
    if callable(task_factory):
        return task_factory()
    return task_factory()


def list_available_tasks() -> List[str]:
    """Return all registered official benchmark task IDs."""
    return sorted(list(BENCHMARK_REGISTRY.keys()))


def get_profile_tasks(profile_name: str) -> List[str]:
    """Retrieve task IDs corresponding to an evaluation profile."""
    p_lower = profile_name.lower().strip()
    if p_lower not in EVALUATION_PROFILES:
        raise ValueError(f"Unknown evaluation profile '{profile_name}'. Available: {list(EVALUATION_PROFILES.keys())}")
    return EVALUATION_PROFILES[p_lower]
