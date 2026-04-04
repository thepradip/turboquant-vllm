"""Memory and performance profiler for quantization benchmarking."""

from __future__ import annotations

import time
import torch
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional


@dataclass
class ProfileResult:
    """Result of a profiling run."""
    operation: str
    elapsed_ms: float
    peak_memory_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    metadata: dict = field(default_factory=dict)


class MemoryProfiler:
    """Profile memory usage and performance of quantization operations."""

    def __init__(self):
        self.results: list[ProfileResult] = []

    @contextmanager
    def profile(self, operation: str, num_tokens: int = 0):
        """Context manager to profile an operation."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / (1024 * 1024)

        start_time = time.perf_counter()

        yield

        elapsed = (time.perf_counter() - start_time) * 1000  # ms

        peak_mem = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        throughput = 0.0
        if num_tokens > 0 and elapsed > 0:
            throughput = num_tokens / (elapsed / 1000)

        result = ProfileResult(
            operation=operation,
            elapsed_ms=elapsed,
            peak_memory_mb=peak_mem,
            throughput_tokens_per_sec=throughput,
        )
        self.results.append(result)

    def compare_memory(
        self,
        fp16_bytes: int,
        quantized_bytes: int,
        label: str = "",
    ) -> dict[str, float]:
        """Compare memory usage between FP16 and quantized."""
        ratio = fp16_bytes / max(quantized_bytes, 1)
        savings = (1 - quantized_bytes / max(fp16_bytes, 1)) * 100

        return {
            "label": label,
            "fp16_mb": fp16_bytes / (1024 * 1024),
            "quantized_mb": quantized_bytes / (1024 * 1024),
            "compression_ratio": ratio,
            "memory_savings_pct": savings,
        }

    def summary(self) -> list[dict]:
        """Get summary of all profiled operations."""
        return [
            {
                "operation": r.operation,
                "elapsed_ms": round(r.elapsed_ms, 3),
                "peak_memory_mb": round(r.peak_memory_mb, 2),
                "throughput_tok_s": round(r.throughput_tokens_per_sec, 1),
            }
            for r in self.results
        ]

    def clear(self) -> None:
        self.results.clear()
