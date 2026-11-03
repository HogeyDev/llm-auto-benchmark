"""
Mathematical Equivalence Evaluator for MATH-500 and Mathematical Benchmarks.
Normalizes LaTeX expressions, simplifies algebraic representations using SymPy,
handles numerical rounding and fractions, and strictly prevents substring credit shortcuts.
"""

import re
from typing import Optional, Union, Tuple
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


def normalize_latex_string(s: str) -> str:
    """Normalize LaTeX mathematical string for canonical comparison."""
    if not s:
        return ""
    text = s.strip()

    # Strip boxed notation
    boxed_match = re.search(r"\\boxed\{([^{}]+|\{[^{}]*\})\}", text)
    if boxed_match:
        text = boxed_match.group(1).strip()

    # Strip dollar signs
    text = text.replace("$", "").strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove formatting commands
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", "").replace(r"\;", "").replace(r"\!", "")
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", text)

    # Normalize fractions \frac{a}{b} -> (a)/(b)
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_pattern.search(text):
        text = frac_pattern.sub(r"((\1)/(\2))", text)

    # Normalize \sqrt{x} -> sqrt(x)
    sqrt_pattern = re.compile(r"\\sqrt\{([^{}]+)\}")
    while sqrt_pattern.search(text):
        text = sqrt_pattern.sub(r"sqrt(\1)", text)

    # Normalize operators
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\pi", "pi")
    text = text.replace("^", "**")

    return text.strip()


def extract_math_answer(response: str) -> Optional[str]:
    """
    Deterministically extract the final mathematical answer from model response.
    Never accepts intermediate reasoning text.
    """
    clean = response.strip()
    if not clean:
        return None

    # 1. Check for \boxed{...} (canonical standard for MATH-500)
    boxed_matches = re.findall(r"\\boxed\{([^{}]+|\{[^{}]*\})\}", clean)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # 2. Check for #### marker
    if "####" in clean:
        after_hash = clean.split("####")[-1].strip()
        lines = [l.strip() for l in after_hash.splitlines() if l.strip()]
        if lines:
            return lines[0]

    # 3. Check for "the answer is ..." or "the final answer is ..."
    final_phrase = re.findall(
        r"(?:the (?:final )?answer is|final result is|the solution is)\s*[:=]?\s*([^\n\.\$]+)",
        clean,
        re.IGNORECASE,
    )
    if final_phrase:
        return final_phrase[-1].strip()

    # 4. Fallback to last line if short and looks like a math expression
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    if lines:
        last = lines[-1].strip().rstrip(".")
        if len(last) < 80 and not any(w in last.lower() for w in ["because", "therefore", "let", "assume", "step", "since"]):
            return last

    return None


def is_math_equivalent(pred: str, target: str) -> bool:
    """
    Strict mathematical equivalence checker between predicted answer and expected target.
    Uses SymPy symbolic equivalence, numerical equivalence, and exact normalized string match.
    If equivalence cannot be safely established, returns False.
    """
    if not pred or not target:
        return False

    norm_pred = normalize_latex_string(pred)
    norm_target = normalize_latex_string(target)

    # 1. Exact string match after LaTeX normalization
    if norm_pred.lower() == norm_target.lower():
        return True

    # Strip surrounding parentheses
    p_core = norm_pred.strip("()")
    t_core = norm_target.strip("()")
    if p_core.lower() == t_core.lower():
        return True

    # 2. Coordinate tuple / list comparison e.g. (3, pi/2)
    if (norm_pred.startswith("(") and norm_pred.endswith(")")) and (norm_target.startswith("(") and norm_target.endswith(")")):
        p_parts = [p.strip() for p in norm_pred[1:-1].split(",")]
        t_parts = [t.strip() for t in norm_target[1:-1].split(",")]
        if len(p_parts) == len(t_parts):
            if all(is_math_equivalent(p, t) for p, t in zip(p_parts, t_parts)):
                return True

    # 3. Floating-point numerical comparison
    try:
        f_pred = float(p_core)
        f_target = float(t_core)
        if abs(f_pred - f_target) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # 4. Symbolic equivalence with SymPy
    try:
        transformations = standard_transformations + (implicit_multiplication_application,)
        sym_pred = parse_expr(norm_pred, transformations=transformations, evaluate=True)
        sym_target = parse_expr(norm_target, transformations=transformations, evaluate=True)

        diff = sympy.simplify(sym_pred - sym_target)
        if diff == 0:
            return True

        # Check numerical equality
        try:
            val = float(diff.evalf())
            if abs(val) < 1e-6:
                return True
        except Exception:
            pass
    except Exception:
        pass

    return False
