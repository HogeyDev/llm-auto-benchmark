"""
Hardware and inference telemetry collector.
Monitors:
- TTFT (Time To First Token in ms)
- Prefill tokens/sec & Decode tokens/sec
- TPOT (Time Per Output Token in ms)
- Inter-token latency statistics (p50, p95, p99)
- Peak system RAM (MB) and peak GPU VRAM (MB)
- Model load time & disk footprint
"""

import time
import statistics
from typing import Optional, List, Dict, Any
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TelemetrySnapshot:
    """Telemetry data captured for a single generation call or evaluation session."""
    def __init__(
        self,
        ttft_ms: Optional[float] = None,
        prefill_tokens_per_sec: Optional[float] = None,
        decode_tokens_per_sec: Optional[float] = None,
        tpot_ms: Optional[float] = None,
        inter_token_latencies_ms: Optional[List[float]] = None,
        peak_vram_mb: Optional[float] = None,
        peak_ram_mb: Optional[float] = None,
        tokens_prompt: int = 0,
        tokens_generated: int = 0,
        total_duration_ms: float = 0.0,
    ):
        self.ttft_ms = ttft_ms
        self.prefill_tokens_per_sec = prefill_tokens_per_sec
        self.decode_tokens_per_sec = decode_tokens_per_sec
        self.tpot_ms = tpot_ms
        self.inter_token_latencies_ms = inter_token_latencies_ms or []
        self.peak_vram_mb = peak_vram_mb
        self.peak_ram_mb = peak_ram_mb
        self.tokens_prompt = tokens_prompt
        self.tokens_generated = tokens_generated
        self.total_duration_ms = total_duration_ms

    def to_dict(self) -> Dict[str, Any]:
        p50 = None
        p95 = None
        if self.inter_token_latencies_ms:
            sorted_lat = sorted(self.inter_token_latencies_ms)
            p50 = statistics.median(sorted_lat)
            idx95 = int(len(sorted_lat) * 0.95)
            p95 = sorted_lat[min(idx95, len(sorted_lat) - 1)]

        return {
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "prefill_tokens_per_sec": round(self.prefill_tokens_per_sec, 2) if self.prefill_tokens_per_sec is not None else None,
            "decode_tokens_per_sec": round(self.decode_tokens_per_sec, 2) if self.decode_tokens_per_sec is not None else None,
            "tpot_ms": round(self.tpot_ms, 2) if self.tpot_ms is not None else None,
            "latency_p50_ms": round(p50, 2) if p50 is not None else None,
            "latency_p95_ms": round(p95, 2) if p95 is not None else None,
            "peak_vram_mb": round(self.peak_vram_mb, 2) if self.peak_vram_mb is not None else None,
            "peak_ram_mb": round(self.peak_ram_mb, 2) if self.peak_ram_mb is not None else None,
            "tokens_prompt": self.tokens_prompt,
            "tokens_generated": self.tokens_generated,
            "total_duration_ms": round(self.total_duration_ms, 2),
        }


class PerformanceCollector:
    """Monitors resource consumption and computes aggregated inference latency metrics."""

    def __init__(self):
        self.process = psutil.Process()
        self.snapshots: List[TelemetrySnapshot] = []
        self._initial_ram_mb = self.get_current_ram_mb()
        self._initial_vram_mb = self.get_current_vram_mb()
        self.peak_vram_observed = self._initial_vram_mb or 0.0
        self.peak_ram_observed = self._initial_ram_mb

    def get_current_ram_mb(self) -> float:
        """Return current RSS memory of the process in MB."""
        try:
            return self.process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def get_current_vram_mb(self) -> Optional[float]:
        """Return currently allocated CUDA VRAM in MB if available."""
        if HAS_TORCH and torch.cuda.is_available():
            try:
                return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            except Exception:
                pass
        return None

    def sample_memory(self) -> None:
        """Sample and update peak observed memory."""
        ram = self.get_current_ram_mb()
        if ram > self.peak_ram_observed:
            self.peak_ram_observed = ram
        vram = self.get_current_vram_mb()
        if vram is not None and vram > self.peak_vram_observed:
            self.peak_vram_observed = vram

    def record_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """Record a generation snapshot and update peak usage."""
        if snapshot.peak_ram_mb and snapshot.peak_ram_mb > self.peak_ram_observed:
            self.peak_ram_observed = snapshot.peak_ram_mb
        if snapshot.peak_vram_mb and snapshot.peak_vram_mb > self.peak_vram_observed:
            self.peak_vram_observed = snapshot.peak_vram_mb
        self.snapshots.append(snapshot)

    def get_aggregate_summary(self) -> Dict[str, Any]:
        """Compute average and peak performance statistics across all recorded calls."""
        if not self.snapshots:
            return {
                "avg_ttft_ms": None,
                "avg_decode_tokens_per_sec": None,
                "avg_tpot_ms": None,
                "peak_vram_mb": round(self.peak_vram_observed, 2) if self.peak_vram_observed > 0 else None,
                "peak_ram_mb": round(self.peak_ram_observed, 2),
                "total_calls": 0,
                "total_tokens_generated": 0,
            }

        ttfts = [s.ttft_ms for s in self.snapshots if s.ttft_ms is not None]
        decodes = [s.decode_tokens_per_sec for s in self.snapshots if s.decode_tokens_per_sec is not None and s.decode_tokens_per_sec > 0]
        tpots = [s.tpot_ms for s in self.snapshots if s.tpot_ms is not None]
        total_tokens = sum(s.tokens_generated for s in self.snapshots)

        return {
            "avg_ttft_ms": round(statistics.mean(ttfts), 2) if ttfts else None,
            "avg_decode_tokens_per_sec": round(statistics.mean(decodes), 2) if decodes else None,
            "avg_tpot_ms": round(statistics.mean(tpots), 2) if tpots else None,
            "peak_vram_mb": round(self.peak_vram_observed, 2) if self.peak_vram_observed > 0 else None,
            "peak_ram_mb": round(self.peak_ram_observed, 2),
            "total_calls": len(self.snapshots),
            "total_tokens_generated": total_tokens,
        }
