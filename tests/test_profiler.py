"""Tests for memory and performance profiler."""

import pytest
import torch
import time
from turboquant.utils.profiler import MemoryProfiler, ProfileResult


class TestProfileResult:
    def test_defaults(self):
        r = ProfileResult(operation="test", elapsed_ms=10.0)
        assert r.operation == "test"
        assert r.elapsed_ms == 10.0
        assert r.peak_memory_mb == 0.0
        assert r.throughput_tokens_per_sec == 0.0
        assert r.metadata == {}


class TestMemoryProfiler:
    def test_profile_captures_timing(self):
        profiler = MemoryProfiler()
        with profiler.profile("encode", num_tokens=100):
            # Simulate work
            _ = torch.randn(100, 128) @ torch.randn(128, 64)

        assert len(profiler.results) == 1
        assert profiler.results[0].operation == "encode"
        assert profiler.results[0].elapsed_ms > 0

    def test_profile_with_tokens_computes_throughput(self):
        profiler = MemoryProfiler()
        with profiler.profile("decode", num_tokens=1000):
            _ = torch.randn(100, 128)

        r = profiler.results[0]
        assert r.throughput_tokens_per_sec > 0

    def test_profile_without_tokens_zero_throughput(self):
        profiler = MemoryProfiler()
        with profiler.profile("load"):
            _ = torch.randn(10, 10)

        assert profiler.results[0].throughput_tokens_per_sec == 0.0

    def test_multiple_profiles(self):
        profiler = MemoryProfiler()
        for i in range(3):
            with profiler.profile(f"op_{i}"):
                _ = torch.randn(10, 10)

        assert len(profiler.results) == 3
        assert profiler.results[0].operation == "op_0"
        assert profiler.results[2].operation == "op_2"

    def test_compare_memory(self):
        profiler = MemoryProfiler()
        result = profiler.compare_memory(
            fp16_bytes=1024 * 1024,  # 1 MB
            quantized_bytes=256 * 1024,  # 0.25 MB
            label="4-bit test",
        )
        assert result["label"] == "4-bit test"
        assert result["fp16_mb"] == 1.0
        assert result["quantized_mb"] == 0.25
        assert result["compression_ratio"] == 4.0
        assert result["memory_savings_pct"] == 75.0

    def test_compare_memory_zero_quantized(self):
        profiler = MemoryProfiler()
        result = profiler.compare_memory(fp16_bytes=1024, quantized_bytes=0)
        assert result["compression_ratio"] == 1024  # fp16 / 1

    def test_summary(self):
        profiler = MemoryProfiler()
        with profiler.profile("encode", num_tokens=500):
            _ = torch.randn(100, 128)
        with profiler.profile("decode", num_tokens=500):
            _ = torch.randn(100, 128)

        summary = profiler.summary()
        assert len(summary) == 2
        assert summary[0]["operation"] == "encode"
        assert "elapsed_ms" in summary[0]
        assert "peak_memory_mb" in summary[0]
        assert "throughput_tok_s" in summary[0]

    def test_clear(self):
        profiler = MemoryProfiler()
        with profiler.profile("test"):
            pass
        assert len(profiler.results) == 1
        profiler.clear()
        assert len(profiler.results) == 0
