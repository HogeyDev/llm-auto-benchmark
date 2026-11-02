"""
Instruction following and strict constraint compliance benchmark task (IFEval-style).
Evaluates verifiable constraints:
- Casing, punctuation, and negative rules
- Markdown structure and count constraints
- Strict programmatic JSON schema validation
"""

import json
import re
from typing import Optional, List, Dict, Any, Callable
from .base import BaseBenchmarkTask
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric


def verify_json_schema(text: str, required_keys: Dict[str, type]) -> bool:
    """Check if the text contains a valid JSON object with the required keys and types."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return False
    try:
        data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return False
        for key, expected_type in required_keys.items():
            if key not in data or not isinstance(data[key], expected_type):
                return False
        return True
    except json.JSONDecodeError:
        return False


def verify_negative_constraint(text: str, forbidden_substr: str) -> bool:
    """Verify that forbidden substring or character is not present in output."""
    return forbidden_substr.lower() not in text.lower()


def verify_bullet_count(text: str, expected_count: int) -> bool:
    """Count bullet points (lines starting with *, -, or numbers)."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    bullets = [line for line in lines if re.match(r"^(\*|-|\+|\d+\.)\s+", line)]
    return len(bullets) == expected_count


def verify_word_count(text: str, min_words: int, max_words: int) -> bool:
    """Verify text length in word counts."""
    words = text.strip().split()
    return min_words <= len(words) <= max_words


def verify_markdown_table(text: str, min_columns: int = 3, min_rows: int = 2) -> bool:
    """Verify text contains a markdown table with header and rows."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
    if len(lines) < min_rows + 2:  # header + separator + rows
        return False
    # Check separator line with dashes
    if not any("---" in l or "-|-" in l for l in lines):
        return False
    header_cols = len([c for c in lines[0].split("|") if c.strip()])
    return header_cols >= min_columns


INSTRUCTION_DATASET = [
    {
        "id": "ifeval_json_france_metadata",
        "prompt": "Provide the country information for France. Respond ONLY with a valid JSON object containing the keys: 'country' (str), 'capital' (str), and 'population_millions' (float or int). Do not include markdown or explanations.",
        "validator": lambda t: verify_json_schema(t, {"country": str, "capital": str, "population_millions": (int, float)}),
        "rule": "Strict JSON schema with typed keys",
    },
    {
        "id": "ifeval_negative_constraint_e",
        "prompt": "Write a short sentence about an animal jumping without using the letter 'e' (neither lowercase 'e' nor uppercase 'E').",
        "validator": lambda t: verify_negative_constraint(t, "e") and len(t.strip()) > 5,
        "rule": "Negative constraint: no letter 'e'",
    },
    {
        "id": "ifeval_exact_three_bullets",
        "prompt": "List exactly 3 benefits of regular exercise. Format each benefit as a separate bullet point starting with a dash '-'. Do not include extra bullets.",
        "validator": lambda t: verify_bullet_count(t, 3),
        "rule": "Format constraint: exactly 3 bullet points",
    },
    {
        "id": "ifeval_word_count_20_30",
        "prompt": "Write a concise summary of the water cycle. Your entire response MUST contain between 20 and 30 words in total.",
        "validator": lambda t: verify_word_count(t, 20, 30),
        "rule": "Length constraint: 20-30 words",
    },
    {
        "id": "ifeval_bookend_words",
        "prompt": "Answer the question: 'Is python an interpreted language?'. Your response MUST start with the exact word 'Certainly' and MUST end with the exact word 'Goodbye.'",
        "validator": lambda t: t.strip().startswith("Certainly") and t.strip().endswith("Goodbye."),
        "rule": "Boundary constraint: start with 'Certainly', end with 'Goodbye.'",
    },
    {
        "id": "ifeval_all_lowercase",
        "prompt": "Explain what an atom is using only lowercase letters. Do not use ANY capital or uppercase letters anywhere in your response.",
        "validator": lambda t: len(t.strip()) > 15 and not any(c.isupper() for c in t),
        "rule": "Casing constraint: all lowercase letters",
    },
    {
        "id": "ifeval_all_caps_warning",
        "prompt": "Write a short security warning about phishing emails. Your entire response MUST be written in ALL CAPITAL LETTERS (uppercase).",
        "validator": lambda t: len(t.strip()) > 15 and not any(c.islower() for c in t) and any(c.isupper() for c in t),
        "rule": "Casing constraint: all uppercase letters",
    },
    {
        "id": "ifeval_markdown_table_3x2",
        "prompt": "Create a Markdown table displaying 3 programming languages with columns: | Language | Paradigm | Typing |. Include at least 2 language rows.",
        "validator": lambda t: verify_markdown_table(t, min_columns=3, min_rows=2),
        "rule": "Structure constraint: 3-column markdown table",
    },
    {
        "id": "ifeval_repeat_phrase_twice",
        "prompt": "Write a statement about software quality. You MUST include the exact phrase 'AI benchmark standard' exactly two times in your response.",
        "validator": lambda t: t.count("AI benchmark standard") == 2,
        "rule": "Keyword repetition: phrase occurs exactly twice",
    },
    {
        "id": "ifeval_no_commas",
        "prompt": "Write a short 3-sentence description of a desert without using any commas anywhere in your text.",
        "validator": lambda t: "," not in t and len(t.strip()) > 20,
        "rule": "Punctuation constraint: zero commas",
    },
    {
        "id": "ifeval_numbered_list_four",
        "prompt": "List four seasons of the year as a numbered list from 1. to 4. Do not include any other text.",
        "validator": lambda t: all(f"{i}." in t for i in range(1, 5)) and "5." not in t,
        "rule": "Numbered list: exactly 4 items",
    },
    {
        "id": "ifeval_json_server_config",
        "prompt": "Generate a server configuration. Respond ONLY with a valid JSON object with keys: 'host' (string), 'port' (integer), 'ssl' (boolean), and 'tags' (array).",
        "validator": lambda t: verify_json_schema(t, {"host": str, "port": int, "ssl": bool, "tags": list}),
        "rule": "Strict JSON schema with typed fields",
    },
    {
        "id": "ifeval_include_three_numbers",
        "prompt": "Write a brief math trivia note that explicitly contains each of the numbers 7, 42, and 100.",
        "validator": lambda t: all(re.search(r"\b" + str(n) + r"\b", t) for n in [7, 42, 100]),
        "rule": "Content inclusion: must include 7, 42, and 100",
    },
    {
        "id": "ifeval_forbidden_words_car",
        "prompt": "Describe an automobile without using any of the following forbidden words: 'drive', 'wheel', 'engine', 'car'.",
        "validator": lambda t: len(t.strip()) > 20 and not any(w in t.lower() for w in ["drive", "wheel", "engine", "car"]),
        "rule": "Multiple negative constraints: forbidden vocabulary",
    },
    {
        "id": "ifeval_exact_two_paragraphs",
        "prompt": "Write an explanation of renewable energy in exactly two paragraphs, separated by a blank line. Do not include a third paragraph.",
        "validator": lambda t: len([p for p in t.strip().split("\n\n") if p.strip()]) == 2,
        "rule": "Paragraph constraint: exactly two paragraphs",
    },
    {
        "id": "ifeval_xml_tag_wrapping",
        "prompt": "Provide your analysis of electric vehicles wrapped strictly inside <analysis> and </analysis> tags. Do not write text outside the tags.",
        "validator": lambda t: t.strip().startswith("<analysis>") and t.strip().endswith("</analysis>"),
        "rule": "Tag constraint: strictly wrapped in XML tags",
    },
    {
        "id": "ifeval_bracketed_citation",
        "prompt": "State a scientific fact about gravity and include a bracketed citation reference '[Ref 1]' at the end.",
        "validator": lambda t: "[Ref 1]" in t,
        "rule": "Citation syntax: contains '[Ref 1]'",
    },
    {
        "id": "ifeval_minimum_length_no_bullets",
        "prompt": "Write a detailed explanation of photosynthesis. It must contain at least 200 characters and MUST NOT contain any bullet points (no *, -, or numbered lists).",
        "validator": lambda t: len(t.strip()) >= 200 and not any(line.strip().startswith(("-", "*", "•")) for line in t.split("\n")),
        "rule": "Length and format: >=200 chars and no bullets",
    },
    {
        "id": "ifeval_title_case_sentence",
        "prompt": "Write a 5-word sentence where every single word begins with an uppercase letter.",
        "validator": lambda t: len(t.strip().split()) == 5 and all(w[0].isupper() for w in t.strip().split() if w[0].isalpha()),
        "rule": "Capitalization: Title Case on all words",
    },
    {
        "id": "ifeval_even_word_count",
        "prompt": "Write a sentence describing the night sky. The total word count of your response MUST be an even number between 10 and 20 words.",
        "validator": lambda t: 10 <= len(t.strip().split()) <= 20 and (len(t.strip().split()) % 2 == 0),
        "rule": "Numerical constraint: even word count between 10 and 20",
    },
]


class InstructionFollowingBenchmark(BaseBenchmarkTask):
    """Evaluates strict instruction following and constraint verification (IFEval-style)."""

    def __init__(self):
        super().__init__(
            task_id="slm_instruction_following",
            category="instruction",
            description="Verifiable instruction adherence, negative constraints, and format compliance (IFEval-style).",
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
        dataset = INSTRUCTION_DATASET[:sample_limit] if sample_limit else INSTRUCTION_DATASET
        total = len(dataset)
        passed = 0
        total_latency = 0.0
        total_tokens = 0
        breakdown = []

        system_prompt = "You are an instruction-following assistant. You must adhere strictly to all user constraints."

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                break
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
            ]
            gen = backend.chat(messages, max_new_tokens=256, temperature=0.0)
            total_latency += gen.latency_ms
            total_tokens += gen.tokens_generated

            is_compliant = False
            try:
                is_compliant = bool(item["validator"](gen.text))
            except Exception:
                is_compliant = False

            if is_compliant:
                passed += 1

            if on_sample:
                on_sample({
                    "task_id": self.task_id,
                    "category": self.category,
                    "sample_index": idx + 1,
                    "total_samples": total,
                    "prompt": item["prompt"],
                    "response": gen.text,
                    "expected": item["rule"],
                    "extracted": "Compliant" if is_compliant else "Failed constraints",
                    "passed": is_compliant,
                    "latency_ms": gen.latency_ms,
                    "tokens": gen.tokens_generated,
                })

            breakdown.append({
                "index": idx,
                "id": item.get("id", str(idx)),
                "rule": item["rule"],
                "passed": is_compliant,
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
            sub_metrics={"sample_details": breakdown},
        )
