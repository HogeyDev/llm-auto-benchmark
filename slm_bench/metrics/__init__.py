"""Metrics and telemetry package for SLM-Bench."""
from .confidence import wilson_score_interval, compute_metric_confidence
from .telemetry import TelemetrySnapshot, PerformanceCollector

__all__ = [
    "wilson_score_interval",
    "compute_metric_confidence",
    "TelemetrySnapshot",
    "PerformanceCollector",
]
