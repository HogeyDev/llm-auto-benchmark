"""
Base classes for SLM benchmark tasks and evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from ..backends.base import BaseInferenceBackend
from ..models.schema import TaskMetric


class BaseBenchmarkTask(ABC):
    """Abstract interface for an individual evaluation task."""

    def __init__(
        self,
        task_id: str,
        category: str,
        description: str,
        raw_metric_name: str = "accuracy",
        weight: float = 1.0,
    ):
        self.task_id = task_id
        self.category = category
        self.description = description
        self.raw_metric_name = raw_metric_name
        self.weight = weight

    @abstractmethod
    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
    ) -> TaskMetric:
        """Execute evaluation samples and return normalized TaskMetric."""
        pass
