"""
Central Benchmark Registry and Evaluation Profiles.
Provides discovery, dynamic task instantiation, and profiles (quick, standard, extended).
"""

from typing import Dict, List, Optional, Type
from .base import BaseBenchmarkTask

# Built-in tasks
from .reasoning_math import ReasoningBenchmark
from .coding_agent import CodingBenchmark
from .instruction_following import InstructionFollowingBenchmark
from .tool_calling import ToolCallingBenchmark

# Expanded benchmark adapters
from .reasoning.mmlu_pro import MMLUProBenchmark
from .reasoning.gpqa import GPQADiamondBenchmark
from .reasoning.bbh import BBHBenchmark
from .mathematics.gsm8k import GSM8KBenchmark
from .mathematics.math500 import MATH500Benchmark
from .coding.evalplus import EvalPlusBenchmark
from .coding.livecodebench import LiveCodeBenchBenchmark
from .instruction.ifbench import IFBenchBenchmark
from .tools.bfcl import BFCLBenchmark
from .context.ruler import RULERBenchmark
from .safety.safety_adapter import SafetyBenchmark
from .external.lm_eval_adapter import LMEvalHarnessAdapter


# Registry mapping task IDs to Task classes
BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmarkTask]] = {
    # Core tasks
    "slm_reasoning_cot": ReasoningBenchmark,
    "slm_coding_agent": CodingBenchmark,
    "slm_instruction_ifeval": InstructionFollowingBenchmark,
    "slm_tool_calling_multiturn": ToolCallingBenchmark,

    # Reasoning & Math
    "mmlu_pro": MMLUProBenchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "bbh": BBHBenchmark,
    "gsm8k": GSM8KBenchmark,
    "math500": MATH500Benchmark,

    # Coding
    "evalplus": EvalPlusBenchmark,
    "evalplus_coding": EvalPlusBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,

    # Instruction & Tools
    "ifbench": IFBenchBenchmark,
    "bfcl": BFCLBenchmark,
    "bfcl_v4": BFCLBenchmark,

    # Long Context & Safety
    "ruler": RULERBenchmark,
    "ruler_context": RULERBenchmark,
    "safety": SafetyBenchmark,
    "safety_reliability": SafetyBenchmark,

    # LM-Eval Harness bridges
    "lm_eval_gsm8k": lambda: LMEvalHarnessAdapter("gsm8k"),
    "lm_eval_mmlu_pro": lambda: LMEvalHarnessAdapter("mmlu_pro"),
}

# Aliases for convenience
TASK_ALIASES: Dict[str, str] = {
    "reasoning": "slm_reasoning_cot",
    "math": "slm_reasoning_cot",
    "coding": "slm_coding_agent",
    "code": "slm_coding_agent",
    "instruction": "slm_instruction_ifeval",
    "ifeval": "slm_instruction_ifeval",
    "tool_calling": "slm_tool_calling_multiturn",
    "tools": "slm_tool_calling_multiturn",
}

# Evaluation Profiles
EVALUATION_PROFILES: Dict[str, List[str]] = {
    "quick": [
        "slm_reasoning_cot",
        "slm_coding_agent",
        "slm_instruction_ifeval",
        "slm_tool_calling_multiturn",
    ],
    "standard": [
        "slm_reasoning_cot",
        "slm_coding_agent",
        "slm_instruction_ifeval",
        "slm_tool_calling_multiturn",
        "mmlu_pro",
        "gsm8k",
        "evalplus",
        "bfcl",
    ],
    "extended": [
        "slm_reasoning_cot",
        "slm_coding_agent",
        "slm_instruction_ifeval",
        "slm_tool_calling_multiturn",
        "mmlu_pro",
        "gpqa_diamond",
        "bbh",
        "gsm8k",
        "math500",
        "evalplus",
        "livecodebench",
        "ifbench",
        "bfcl",
        "ruler",
        "safety",
    ],
}


def get_task(task_name_or_id: str) -> BaseBenchmarkTask:
    """Retrieve and instantiate a benchmark task by ID or alias."""
    task_id = TASK_ALIASES.get(task_name_or_id.lower(), task_name_or_id)
    if task_id not in BENCHMARK_REGISTRY:
        available = ", ".join(list(BENCHMARK_REGISTRY.keys()) + list(TASK_ALIASES.keys()))
        raise ValueError(f"Unknown benchmark task '{task_name_or_id}'. Available tasks: {available}")
    cls_or_fn = BENCHMARK_REGISTRY[task_id]
    return cls_or_fn()


def list_available_tasks() -> List[str]:
    """Return all registered unique task identifiers."""
    return sorted(list(BENCHMARK_REGISTRY.keys()))


def get_profile_tasks(profile_name: str) -> List[str]:
    """Retrieve list of task IDs associated with an evaluation profile."""
    prof = profile_name.lower().strip()
    if prof not in EVALUATION_PROFILES:
        raise ValueError(f"Unknown evaluation profile '{profile_name}'. Choose from: {list(EVALUATION_PROFILES.keys())}")
    return EVALUATION_PROFILES[prof]
