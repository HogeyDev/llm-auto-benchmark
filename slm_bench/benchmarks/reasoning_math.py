"""
Multi-step mathematical reasoning and ARC-Challenge benchmark task.
Evaluates:
- GSM-Symbolic multi-step arithmetic reasoning with varied parameters
- ARC-Challenge multi-hop scientific deduction and commonsense logic
"""

import re
from typing import Optional, List, Dict, Any
from .base import BaseBenchmarkTask
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric
from ..utils.logger import logger


REASONING_DATASET = [
    # ─── GSM-Symbolic Multi-Step Arithmetic (15 Items) ────────────────────────
    {
        "id": "gsm_water_tank_rate",
        "type": "numeric",
        "question": "A water tank holds 500 liters. A pump fills it at 40 liters per minute while a drain empties it at 15 liters per minute. If the tank is initially empty, how many minutes will it take to fill completely?",
        "expected": "20",
    },
    {
        "id": "gsm_train_average_speed",
        "type": "numeric",
        "question": "A train travels at 75 km/h for 2 hours, and then at 90 km/h for 3 hours. What is the average speed of the train over the entire 5-hour journey in km/h?",
        "expected": "84",
    },
    {
        "id": "gsm_chickens_and_cows",
        "type": "numeric",
        "question": "A farmer has chickens and cows. In total there are 30 heads and 88 legs. How many cows does the farmer have?",
        "expected": "14",
    },
    {
        "id": "gsm_laptop_discount_chain",
        "type": "numeric",
        "question": "A laptop costs $800. The store increases the price by 20%, and then offers a 20% discount on the new price during a holiday sale. What is the final sale price of the laptop in dollars?",
        "expected": "768",
    },
    {
        "id": "gsm_cookie_batches_revenue",
        "type": "numeric",
        "question": "A bakery bakes 15 batches of cookies with 24 cookies per batch. They pack them into boxes of 8 cookies each and sell each box for $6. If all boxes are sold, how much revenue in dollars did the bakery generate?",
        "expected": "270",
    },
    {
        "id": "gsm_round_trip_average_speed",
        "type": "numeric",
        "question": "A car travels from City A to City B at 60 mph and returns along the same route at 40 mph. If the distance between the cities is 120 miles, what is the car's average speed for the entire round trip in mph?",
        "expected": "48",
    },
    {
        "id": "gsm_consecutive_integers",
        "type": "numeric",
        "question": "The sum of three consecutive integers is 72. What is the largest of these three integers?",
        "expected": "25",
    },
    {
        "id": "gsm_runner_speed_distance",
        "type": "numeric",
        "question": "A runner completes a 400-meter lap in 50 seconds. At this rate, how many meters can the runner complete in 3 minutes?",
        "expected": "1440",
    },
    {
        "id": "gsm_factory_production_shift",
        "type": "numeric",
        "question": "A factory produces 450 items on Monday. Production increases by 10% on Tuesday, and drops by 50 items on Wednesday. How many items were produced on Wednesday?",
        "expected": "445",
    },
    {
        "id": "gsm_rectangular_garden_area",
        "type": "numeric",
        "question": "A rectangular garden has a perimeter of 56 meters. Its length is 4 meters greater than its width. What is the area of the garden in square meters?",
        "expected": "192",
    },
    {
        "id": "gsm_machine_packaging_scaling",
        "type": "numeric",
        "question": "If 5 machines can package 100 boxes in 10 minutes, how many boxes can 8 machines package in 15 minutes at the same packaging rate?",
        "expected": "240",
    },
    {
        "id": "gsm_cylinder_capacity_fraction",
        "type": "numeric",
        "question": "A cylindrical container holds 180 ml of liquid when filled to 75% capacity. What is the total volume capacity of the container in ml?",
        "expected": "240",
    },
    {
        "id": "gsm_age_ratio_word_problem",
        "type": "numeric",
        "question": "Alice is twice as old as Bob. In 6 years, the sum of their ages will be 48. How old is Alice now?",
        "expected": "24",
    },
    {
        "id": "gsm_fruit_ratio_percentage",
        "type": "numeric",
        "question": "A store sells 40 apples and 60 oranges. A new shipment arrives containing 20 apples and 30 oranges. What percentage (0-100) of the total fruit inventory are apples?",
        "expected": "40",
    },
    {
        "id": "gsm_baker_cakes_wedding",
        "type": "numeric",
        "question": "A baker makes 12 cakes per hour. He bakes for 4 hours. Later, he sells 35 cakes to a wedding party. How many cakes does he have left?",
        "expected": "13",
    },

    # ─── ARC-Challenge Scientific Deduction & Logic (10 Items) ────────────────
    {
        "id": "arc_mineral_scratch_hardness",
        "type": "multiple_choice",
        "question": "Which property of a mineral can be tested by scratching it with a copper penny?\n(A) color\n(B) streak\n(C) hardness\n(D) luster",
        "expected": "C",
    },
    {
        "id": "arc_solar_oven_thermal_absorption",
        "type": "multiple_choice",
        "question": "An engineer wants to design a solar oven that will heat food quickly. Which color should the interior of the solar oven be painted to absorb the most thermal energy?\n(A) white\n(B) silver\n(C) black\n(D) yellow",
        "expected": "C",
    },
    {
        "id": "arc_rubber_ball_newton_third_law",
        "type": "multiple_choice",
        "question": "A student throws a rubber ball against a brick wall. The ball bounces back. Which law of motion best explains why the ball bounces back?\n(A) Newton's first law\n(B) Newton's second law\n(C) Newton's third law\n(D) Kepler's third law",
        "expected": "C",
    },
    {
        "id": "arc_phase_change_condensation",
        "type": "multiple_choice",
        "question": "During which phase change do water molecules lose kinetic energy and move closer together?\n(A) melting\n(B) evaporation\n(C) sublimation\n(D) condensation",
        "expected": "D",
    },
    {
        "id": "arc_marine_fossil_mountain_peak",
        "type": "multiple_choice",
        "question": "A scientist discovers a fossil of a marine fish on a mountain peak. What does this discovery most likely indicate?\n(A) The fish could survive on dry land\n(B) The mountain was once covered by an ocean\n(C) Birds carried the fish to the mountaintop\n(D) Ancient humans placed the fossil there",
        "expected": "B",
    },
    {
        "id": "arc_plant_cell_chloroplast",
        "type": "multiple_choice",
        "question": "Which cellular structure is found in plant cells but NOT in animal cells?\n(A) cell membrane\n(B) chloroplast\n(C) mitochondrion\n(D) nucleus",
        "expected": "B",
    },
    {
        "id": "arc_circuit_energy_transformation",
        "type": "multiple_choice",
        "question": "A light bulb is connected to a battery in a complete circuit. What energy transformation takes place in the light bulb?\n(A) Electrical energy to light and thermal energy\n(B) Chemical energy to electrical energy\n(C) Mechanical energy to electrical energy\n(D) Thermal energy to chemical energy",
        "expected": "A",
    },
    {
        "id": "arc_sound_wave_frequency_pitch",
        "type": "multiple_choice",
        "question": "If the frequency of a sound wave increases, what happens to the perceived pitch of the sound?\n(A) The pitch becomes lower\n(B) The pitch becomes higher\n(C) The pitch remains unchanged\n(D) The sound becomes inaudible",
        "expected": "B",
    },
    {
        "id": "arc_renewable_energy_solar",
        "type": "multiple_choice",
        "question": "Which of the following is a renewable natural resource?\n(A) Coal\n(B) Natural gas\n(C) Solar energy\n(D) Uranium",
        "expected": "C",
    },
    {
        "id": "arc_transform_fault_earthquake",
        "type": "multiple_choice",
        "question": "Two tectonic plates slide horizontally past one another along a transform fault. Which geological event is most likely to result from this motion?\n(A) Volcanic island arc formation\n(B) Earthquakes\n(C) Deep ocean trenches\n(D) Folded mountain ranges",
        "expected": "B",
    },
]


def extract_reasoning_answer(text: str, item_type: str = "numeric") -> Optional[str]:
    """
    Extract answer from chain-of-thought text.
    Handles numeric GSM problems and multiple-choice ARC questions.
    """
    clean_text = text.strip()

    if item_type == "multiple_choice":
        # Look for #### <letter>
        if "####" in clean_text:
            ans_part = clean_text.split("####")[-1].strip()
            match = re.search(r"\b([A-D])\b", ans_part, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Look for \boxed{A}
        boxed_match = re.search(r"\\boxed\{\s*([A-D])\s*\}", clean_text, re.IGNORECASE)
        if boxed_match:
            return boxed_match.group(1).upper()

        # Look for "the answer is (A)" or "the answer is A"
        phrase_match = re.findall(r"(?:answer is|correct choice is|option is|answer:)\s*\(?([A-D])\)?", clean_text, re.IGNORECASE)
        if phrase_match:
            return phrase_match[-1].upper()

        # Look for final standalone (A) or A. at the end of the text
        standalone_match = re.findall(r"(?:^|\s|\()([A-D])(?:\)|\.|\s|$)", clean_text)
        if standalone_match:
            return standalone_match[-1].upper()

        return None

    # Default numeric extraction (GSM-Symbolic)
    # Look for \boxed{<num>}
    boxed_num = re.search(r"\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", clean_text)
    if boxed_num:
        return boxed_num.group(1).rstrip(".")

    # Look for standard GSM #### delimiter
    if "####" in clean_text:
        ans_part = clean_text.split("####")[-1].strip()
        num_match = re.search(r"[-+]?\d+(?:\.\d+)?", ans_part)
        if num_match:
            return num_match.group(0).rstrip(".")

    # Check for phrases like "answer is X"
    match = re.findall(
        r"(?:answer is|is equal to|leaves|speed =|remaining:?|total:?|sale price:?|revenue:?|area:?|\$)\s*([-+]?\d+(?:\.\d+)?)",
        clean_text,
        re.IGNORECASE,
    )
    if match:
        return match[-1].rstrip(".")

    # Fallback to the very last number mentioned in the text
    all_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", clean_text)
    if all_nums:
        return all_nums[-1].rstrip(".")

    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    """Backwards-compatible wrapper for numeric answer extraction."""
    return extract_reasoning_answer(text, item_type="numeric")


class ReasoningBenchmark(BaseBenchmarkTask):
    """Evaluates multi-step mathematical and scientific reasoning (GSM-Symbolic & ARC-Challenge)."""

    def __init__(self):
        super().__init__(
            task_id="slm_reasoning_cot",
            category="reasoning",
            description="Multi-step chain-of-thought mathematical and scientific reasoning (GSM-Symbolic & ARC-Challenge).",
            raw_metric_name="accuracy",
            weight=1.0,
        )

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> TaskMetric:
        dataset = REASONING_DATASET[:sample_limit] if sample_limit else REASONING_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        results_breakdown = []

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                break

            q_type = item.get("type", "numeric")
            if q_type == "multiple_choice":
                system_prompt = (
                    "You are a helpful science and reasoning assistant. "
                    "Analyze the question carefully. At the very end of your response, write '#### <letter>' "
                    "with the single correct option letter (A, B, C, or D)."
                )
                user_msg = f"{item['question']}\n\nSelect the single correct option:"
            else:
                system_prompt = (
                    "You are a helpful mathematical reasoning assistant. "
                    "Think step by step carefully. At the very end of your response, write '#### <number>' "
                    "with the final numeric answer."
                )
                user_msg = f"Solve this problem: {item['question']}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]

            gen = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            extracted = extract_reasoning_answer(gen.text, item_type=q_type)
            expected = item["expected"]

            # Compare answers
            is_correct = False
            if extracted is not None:
                if q_type == "multiple_choice":
                    is_correct = (extracted.strip().upper() == expected.strip().upper())
                else:
                    try:
                        is_correct = abs(float(extracted) - float(expected)) < 1e-4
                    except ValueError:
                        is_correct = (extracted.strip() == expected.strip())

            if is_correct:
                passed += 1

            if on_sample:
                on_sample({
                    "task_id": self.task_id,
                    "category": self.category,
                    "sample_index": idx + 1,
                    "total_samples": total,
                    "prompt": user_msg,
                    "response": gen.text,
                    "expected": expected,
                    "extracted": extracted,
                    "passed": is_correct,
                    "latency_ms": gen.latency_ms,
                    "tokens": gen.tokens_generated,
                })

            results_breakdown.append({
                "index": idx,
                "id": item.get("id", str(idx)),
                "expected": expected,
                "extracted": extracted,
                "passed": is_correct,
            })

        acc = (passed / total) if total > 0 else 0.0
        norm_score = acc * 100.0
        avg_lat = (total_latency / total) if total > 0 else 0.0
        avg_tps = (total_tokens / (total_latency / 1000.0)) if total_latency > 0 else 0.0

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            raw_score=round(acc, 4),
            raw_metric_name=self.raw_metric_name,
            normalized_score=round(norm_score, 2),
            total_samples=total,
            passed_samples=passed,
            avg_latency_ms=round(avg_lat, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={"sample_details": results_breakdown},
        )
