"""
Standardized JSON Schema definitions for SLM Benchmark Results.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class HardwareInfo(BaseModel):
    """Host environment and hardware specifications."""
    os: str = Field(..., description="Operating system name and version")
    cpu: str = Field(..., description="Processor name")
    cpu_cores: int = Field(..., description="Logical CPU cores count")
    total_ram_gb: float = Field(..., description="Total system RAM in GB")
    model_ram_gb: Optional[float] = Field(None, description="Size in RAM / VRAM in GB that the model takes up")
    gpu_name: Optional[str] = Field(None, description="Primary GPU model name")
    gpu_count: int = Field(0, description="Number of detected GPUs")
    gpu_vram_gb: Optional[float] = Field(None, description="Primary GPU VRAM in GB")
    cuda_version: Optional[str] = Field(None, description="Detected CUDA runtime version")
    driver_version: Optional[str] = Field(None, description="GPU Driver version")


class ModelMetadata(BaseModel):
    """Model identity, parameter estimates, and architecture details."""
    model_id: str = Field(..., description="Hugging Face repository ID or model path")
    model_slug: str = Field(..., description="URL and branch safe slug")
    param_count_b: Optional[float] = Field(None, description="Estimated/retrieved parameters in billions")
    param_tier: str = Field(..., description="Parameter tier: <3B, 3B-8B, 8B-14B, 14B-35B, 35B-50B, >50B")
    architecture: Optional[str] = Field(None, description="Model architecture type")
    quantization: str = Field("BF16", description="Quantization method (e.g., GGUF_Q4_K_M, AWQ_4BIT, NF4, BF16)")
    model_ram_gb: Optional[float] = Field(None, description="Size in RAM / VRAM in GB that the model takes up")
    context_length: Optional[int] = Field(None, description="Maximum context window in tokens")
    license: Optional[str] = Field(None, description="Model license")
    backend: str = Field(..., description="Inference backend used for evaluation")


class TaskMetric(BaseModel):
    """Detailed score for an individual benchmark task."""
    task_id: str = Field(..., description="Unique task identifier")
    category: str = Field(..., description="Category: reasoning, coding, instruction, tool_calling")
    raw_score: float = Field(..., description="Raw metric value")
    raw_metric_name: str = Field("accuracy", description="Metric type: accuracy, pass@1, compliance_rate, etc.")
    normalized_score: float = Field(..., ge=0.0, le=100.0, description="Normalized score 0.0 - 100.0%")
    total_samples: int = Field(..., description="Number of test items evaluated")
    passed_samples: int = Field(..., description="Number of passed test items")
    avg_latency_ms: Optional[float] = Field(None, description="Average response latency in milliseconds")
    avg_tokens_per_sec: Optional[float] = Field(None, description="Average generation speed in tokens/second")
    sub_metrics: Dict[str, Any] = Field(default_factory=dict, description="Fine-grained metric breakdowns")


class CategoryScore(BaseModel):
    """Aggregated score for a high-level benchmark pillar."""
    category: str
    weight: float = Field(..., description="Weight in the composite index (0.0 - 1.0)")
    normalized_score: float = Field(..., ge=0.0, le=100.0, description="Average score 0.0 - 100.0%")
    task_count: int


class BenchmarkResult(BaseModel):
    """Standardized root JSON result structure for a benchmarked model."""
    schema_version: str = Field("1.0.0", description="Schema version identifier")
    evaluation_id: str = Field(..., description="Unique UUID for evaluation run")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: ModelMetadata
    hardware: HardwareInfo
    tasks: Dict[str, TaskMetric] = Field(..., description="Task-level evaluation results")
    categories: Dict[str, CategoryScore] = Field(..., description="Category-level aggregates")
    slm_index: float = Field(..., ge=0.0, le=100.0, description="Overall normalized aggregate composite score (0-100%)")
    notes: Optional[str] = Field(None, description="Evaluation run notes or environment details")

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact summary suitable for leaderboard table rendering."""
        return {
            "model_id": self.model.model_id,
            "model_slug": self.model.model_slug,
            "params_b": self.model.param_count_b,
            "param_tier": self.model.param_tier,
            "quantization": self.model.quantization,
            "backend": self.model.backend,
            "slm_index": round(self.slm_index, 2),
            "reasoning": round(self.categories.get("reasoning", CategoryScore(category="reasoning", weight=0.25, normalized_score=0.0, task_count=0)).normalized_score, 2),
            "coding": round(self.categories.get("coding", CategoryScore(category="coding", weight=0.30, normalized_score=0.0, task_count=0)).normalized_score, 2),
            "instruction": round(self.categories.get("instruction", CategoryScore(category="instruction", weight=0.25, normalized_score=0.0, task_count=0)).normalized_score, 2),
            "tool_calling": round(self.categories.get("tool_calling", CategoryScore(category="tool_calling", weight=0.20, normalized_score=0.0, task_count=0)).normalized_score, 2),
            "gpu": self.hardware.gpu_name or "CPU",
            "model_ram_gb": self.hardware.model_ram_gb or self.model.model_ram_gb,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }
