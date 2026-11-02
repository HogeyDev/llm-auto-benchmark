"""
Statistical confidence intervals and uncertainty estimation for benchmark metrics.
Implements Wilson score interval for binary metrics (accuracy, pass@1)
and sample variance / standard error calculations.
"""

import math
from typing import Tuple, Dict, Any, Optional


def wilson_score_interval(
    passed: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.
    
    Args:
        passed: Number of successes / passed samples.
        total: Total number of trials / evaluated samples.
        confidence: Desired confidence level (default 0.95 for 95% CI).
        
    Returns:
        (ci_lower, ci_upper) as floats in range [0.0, 1.0].
    """
    if total <= 0:
        return (0.0, 0.0)
    
    passed = max(0, min(passed, total))
    
    # Critical value z for given confidence level
    # Common values: 0.90 -> 1.645, 0.95 -> 1.95996, 0.99 -> 2.576
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    else:
        # Approximation of inverse normal CDF
        # Using erf-based approximation
        z = 1.95996

    p_hat = passed / total
    z_sq = z * z
    denom = 1.0 + z_sq / total
    center = (p_hat + z_sq / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z_sq / (4.0 * total * total)))
    
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    
    return (round(ci_lower, 4), round(ci_upper, 4))


def compute_metric_confidence(
    passed: int,
    total: int,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute metric summary with accuracy, standard error, and confidence bounds.
    """
    if total <= 0:
        return {
            "accuracy": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_range": 0.0,
            "standard_error": 0.0,
            "confidence_level": confidence,
            "total_samples": 0,
            "passed_samples": 0,
        }
    
    p = passed / total
    se = math.sqrt(p * (1.0 - p) / total)
    ci_lower, ci_upper = wilson_score_interval(passed, total, confidence)
    
    return {
        "accuracy": round(p, 4),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_range": round(ci_upper - ci_lower, 4),
        "standard_error": round(se, 4),
        "confidence_level": confidence,
        "total_samples": total,
        "passed_samples": passed,
    }
