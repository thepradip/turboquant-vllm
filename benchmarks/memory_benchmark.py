"""Memory and throughput benchmark for KV cache quantization.

Measures:
1. Memory usage at various sequence lengths
2. Encode/decode throughput (tokens/sec)
3. Comparison against FP16 baseline
4. Memory savings vs quality trade-off curves
"""

from __future__ import annotations

import json
import time
import torch
from dataclasses import dataclass, asdict

from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer
from turboquant.core.kv_cache import QuantizedKVCache
from turboquant.utils.profiler import MemoryProfiler


@dataclass
class MemoryBenchmarkResult:
    config_name: str
    seq_len: int
    num_layers: int
    # Memory
    fp16_mb: float
    quantized_mb: float
    compression_ratio: float
    memory_savings_pct: float
    # Throughput
    encode_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0


class MemoryBenchmark:
    """Benchmark memory usage and throughput across configurations."""

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        batch_size: int = 1,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size

    def run_all(
        self,
        seq_lengths: list[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768],
    ) -> list[MemoryBenchmarkResult]:
        """Run memory benchmarks for all presets at various sequence lengths."""
        configs = {
            "FP16 Baseline": None,
            "TurboQuant-4bit": TurboQuantConfig.turbo_4bit(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            ),
            "KIVI-2bit": TurboQuantConfig.kivi_2bit(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            ),
            "Bonsai-1bit": TurboQuantConfig.onebit_extreme(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            ),
        }

        results = []
        for name, config in configs.items():
            for seq_len in seq_lengths:
                if name == "FP16 Baseline":
                    result = self._benchmark_fp16(seq_len)
                else:
                    result = self._benchmark_quantized(name, config, seq_len)
                results.append(result)

        return results

    def _fp16_memory_mb(self, seq_len: int) -> float:
        """Calculate FP16 KV cache memory in MB."""
        # 2 (K+V) * batch * seq * kv_heads * head_dim * 2 (bytes) * layers
        total_bytes = (
            2 * self.batch_size * seq_len * self.num_kv_heads
            * self.head_dim * 2 * self.num_layers
        )
        return total_bytes / (1024 * 1024)

    def _benchmark_fp16(self, seq_len: int) -> MemoryBenchmarkResult:
        mb = self._fp16_memory_mb(seq_len)
        return MemoryBenchmarkResult(
            config_name="FP16 Baseline",
            seq_len=seq_len,
            num_layers=self.num_layers,
            fp16_mb=mb,
            quantized_mb=mb,
            compression_ratio=1.0,
            memory_savings_pct=0.0,
        )

    def _benchmark_quantized(
        self, name: str, config: TurboQuantConfig, seq_len: int
    ) -> MemoryBenchmarkResult:
        """Benchmark a quantized configuration at given sequence length."""
        quantizer = TurboQuantizer(config)

        fp16_mb = self._fp16_memory_mb(seq_len)

        # Measure encode throughput
        keys = torch.randn(self.batch_size, seq_len, self.num_kv_heads, self.head_dim)
        values = torch.randn(self.batch_size, seq_len, self.num_kv_heads, self.head_dim)

        start = time.perf_counter()
        q_k, meta_k = quantizer.encode_keys(keys)
        q_v, meta_v = quantizer.encode_values(values)
        encode_time = time.perf_counter() - start
        encode_tps = seq_len / max(encode_time, 1e-6)

        # Measure decode throughput
        start = time.perf_counter()
        quantizer.decode_keys(q_k, meta_k)
        quantizer.decode_values(q_v, meta_v)
        decode_time = time.perf_counter() - start
        decode_tps = seq_len / max(decode_time, 1e-6)

        # Estimate quantized memory
        comp = quantizer.compute_compression_ratio(int(fp16_mb * 1024 * 1024))
        quantized_mb = comp["compressed_bytes"] / (1024 * 1024)

        return MemoryBenchmarkResult(
            config_name=name,
            seq_len=seq_len,
            num_layers=self.num_layers,
            fp16_mb=fp16_mb,
            quantized_mb=quantized_mb,
            compression_ratio=comp["compression_ratio"],
            memory_savings_pct=comp["memory_savings_pct"],
            encode_tokens_per_sec=encode_tps,
            decode_tokens_per_sec=decode_tps,
        )

    def print_results(self, results: list[MemoryBenchmarkResult]) -> None:
        print("\n" + "=" * 100)
        print("MEMORY BENCHMARK RESULTS")
        print("=" * 100)
        print(f"{'Config':<20} {'Seq Len':>8} {'FP16 MB':>8} {'Quant MB':>9} "
              f"{'Ratio':>6} {'Saved':>6} {'Enc tok/s':>10} {'Dec tok/s':>10}")
        print("-" * 100)
        for r in results:
            print(f"{r.config_name:<20} {r.seq_len:>8} {r.fp16_mb:>8.1f} "
                  f"{r.quantized_mb:>9.1f} {r.compression_ratio:>5.1f}x "
                  f"{r.memory_savings_pct:>5.1f}% {r.encode_tokens_per_sec:>10.0f} "
                  f"{r.decode_tokens_per_sec:>10.0f}")

    def save_results(self, results: list[MemoryBenchmarkResult], path: str) -> None:
        data = [asdict(r) for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def main():
    print("TurboQuant Memory & Throughput Benchmark")
    print("=" * 60)

    bench = MemoryBenchmark(
        num_layers=32, num_heads=32, num_kv_heads=8, head_dim=128
    )

    results = bench.run_all(seq_lengths=[512, 1024, 4096, 16384, 32768])
    bench.print_results(results)
    bench.save_results(results, "memory_benchmark_results.json")


if __name__ == "__main__":
    main()
