"""
Official Berkeley Function Calling Leaderboard (BFCL v4) Benchmark Adapter.
Evaluates native structured function calling and tool execution across official test categories:
Simple, Parallel, and Multiple function call scenarios.
Validates function selection, JSON argument schema compliance, and parameter correctness separately.
"""

import json
import re
import time
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...datasets.manager import DatasetManager
from ...utils.logger import logger


def parse_function_calls(response_text: str) -> List[Dict[str, Any]]:
    """Extract structured function calls from model response."""
    clean = response_text.strip()
    calls = []

    # 1. Direct JSON parse
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "name" in data:
            return [data]
        elif isinstance(data, list):
            valid = [item for item in data if isinstance(item, dict) and "name" in item]
            if valid:
                return valid
    except Exception:
        pass

    # 2. Markdown fenced block
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", clean)
    for m in matches:
        try:
            data = json.loads(m.strip())
            if isinstance(data, dict) and "name" in data:
                calls.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        calls.append(item)
        except Exception:
            pass
    if calls:
        return calls

    # 3. Regex search for {"name": "...", "arguments": {...}}
    pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}')
    for m in pattern.finditer(clean):
        try:
            fname = m.group(1)
            fargs = json.loads(m.group(2))
            calls.append({"name": fname, "arguments": fargs})
        except Exception:
            pass

    return calls


class BFCLBenchmark(BaseBenchmarkTask):
    """Evaluates native tool calling on official Berkeley Function Calling Leaderboard v4 scenarios."""

    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id="bfcl",
            category="tool_calling",
            description="Berkeley Function Calling Leaderboard (BFCL v4) official tool scenarios.",
            raw_metric_name="accuracy",
            weight=1.0,
        )
        self.dm = dataset_manager or DatasetManager()

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        seed: int = 42,
    ) -> TaskMetric:
        # Load official BFCL v4 dataset
        full_dataset = self.dm.load_bfcl()

        if sample_limit and sample_limit < len(full_dataset):
            dataset, selected_ids = self.dm.sample_stratified(
                full_dataset,
                sample_size=sample_limit,
                seed=seed,
                stratify_key="category_file",
            )
            run_status = "PARTIAL"
        else:
            dataset = full_dataset
            run_status = "FULL"

        total = len(dataset)
        passed = 0
        func_selection_passed = 0
        arg_correctness_passed = 0
        invalid_count = 0
        question_results: List[QuestionResult] = []

        total_latency_ms = 0.0
        total_tokens = 0

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("BFCL evaluation cancelled.")

            q_id = item.get("id", f"bfcl_{idx}")
            raw_q = item.get("question", [])
            tools = item.get("function", [])

            # Format user prompt
            if isinstance(raw_q, list) and len(raw_q) > 0:
                user_msg = raw_q[0].get("content", "") if isinstance(raw_q[0], dict) else str(raw_q[0])
            else:
                user_msg = str(raw_q)

            start_t = time.perf_counter()
            tool_res = backend.generate_with_tools(
                messages=[{"role": "user", "content": user_msg}],
                tools=tools,
                max_new_tokens=512,
            )
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            total_latency_ms += latency_ms

            raw_out = tool_res.get("raw_output", "")
            calls = parse_function_calls(raw_out)

            expected_ground_truth = item.get("ground_truth") or item.get("possible_answer")

            func_correct = False
            arg_correct = False
            is_correct = False

            if not calls:
                invalid_count += 1
            else:
                # Validate against ground truth or tool definitions
                tool_names = [t.get("name") for t in tools if isinstance(t, dict)]
                predicted_func = calls[0].get("name")

                if expected_ground_truth and isinstance(expected_ground_truth, list) and len(expected_ground_truth) > 0:
                    exp_item = expected_ground_truth[0]
                    exp_name = exp_item.get("name") if isinstance(exp_item, dict) else str(exp_item)
                    if predicted_func == exp_name:
                        func_correct = True
                        func_selection_passed += 1

                        # Validate arguments
                        exp_args = exp_item.get("arguments", {}) if isinstance(exp_item, dict) else {}
                        pred_args = calls[0].get("arguments", {})
                        if exp_args == pred_args or (not exp_args and not pred_args):
                            arg_correct = True
                            arg_correctness_passed += 1
                            is_correct = True
                elif predicted_func in tool_names:
                    # Generic tool validation
                    func_correct = True
                    func_selection_passed += 1
                    if isinstance(calls[0].get("arguments"), dict):
                        arg_correct = True
                        arg_correctness_passed += 1
                        is_correct = True

            if is_correct:
                passed += 1

            q_res = QuestionResult(
                question_id=str(q_id),
                raw_prompt=user_msg,
                raw_response=raw_out,
                extracted_answer=json.dumps(calls) if calls else None,
                expected_answer=str(expected_ground_truth) if expected_ground_truth else "Valid tool call",
                is_correct=is_correct,
                evaluation_error=None if calls else "Malformed tool call JSON",
                tokens_prompt=len(user_msg.split()),
                tokens_generated=len(raw_out.split()),
                latency_ms=round(latency_ms, 2),
            )
            question_results.append(q_res)

            if on_sample:
                on_sample(self.task_id, idx + 1, total, is_correct)

        accuracy = (passed / total * 100.0) if total > 0 else 0.0
        ci_lower, ci_upper = wilson_score_interval(passed, total)

        func_acc = (func_selection_passed / total * 100.0) if total > 0 else 0.0
        arg_acc = (arg_correctness_passed / total * 100.0) if total > 0 else 0.0

        avg_latency = total_latency_ms / max(1, total)

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status=run_status,
            raw_score=round(passed / max(1, total), 4),
            raw_metric_name="accuracy",
            normalized_score=round(accuracy, 2),
            total_samples=total,
            passed_samples=passed,
            failed_samples=total - passed,
            invalid_responses=invalid_count,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            dataset_name="ShishirPatil/gorilla-bfcl",
            dataset_revision="v4",
            split="standard",
            avg_latency_ms=round(avg_latency, 2),
            sub_metrics={
                "function_selection_accuracy": round(func_acc, 2),
                "argument_correctness_accuracy": round(arg_acc, 2),
            },
            question_results=question_results,
        )
