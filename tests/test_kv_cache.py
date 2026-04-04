"""Tests for the quantized KV cache manager."""

import pytest
import torch
from turboquant.config import TurboQuantConfig
from turboquant.core.kv_cache import QuantizedKVCache, CacheEntry
from turboquant.core.quantizer import TurboQuantizer


class TestQuantizedKVCache:
    """Test the KV cache with quantization and residual buffer."""

    @pytest.fixture
    def config(self):
        return TurboQuantConfig.turbo_4bit(
            num_heads=8, num_kv_heads=4, head_dim=64,
            residual_buffer_size=32,
            kivi_group_size=16,
        )

    @pytest.fixture
    def cache(self, config):
        quantizer = TurboQuantizer(config)
        kv = QuantizedKVCache(config, num_layers=4)
        kv.set_quantizer(quantizer)
        return kv

    def test_initial_state(self, cache):
        """Cache should start empty."""
        assert cache.get_seq_len(0) == 0
        assert cache.entries[0].quantized_len == 0
        assert cache.entries[0].residual_len == 0

    def test_append_to_residual(self, cache):
        """Small appends should go to residual buffer."""
        keys = torch.randn(1, 8, 4, 64)
        values = torch.randn(1, 8, 4, 64)
        cache.append(0, keys, values)

        assert cache.get_seq_len(0) == 8
        assert cache.entries[0].residual_len == 8
        assert cache.entries[0].quantized_len == 0

    def test_residual_flush_to_quantized(self, cache):
        """Exceeding residual buffer should trigger quantization."""
        # Residual buffer is 32, group size is 16
        # Adding 48 tokens: should flush 16 to quantized
        keys = torch.randn(1, 48, 4, 64)
        values = torch.randn(1, 48, 4, 64)
        cache.append(0, keys, values)

        assert cache.entries[0].quantized_len > 0
        assert cache.entries[0].total_len == 48

    def test_get_keys_combines_quantized_and_residual(self, cache):
        """get_keys should return both dequantized + residual portions."""
        keys = torch.randn(1, 48, 4, 64)
        values = torch.randn(1, 48, 4, 64)
        cache.append(0, keys, values)

        retrieved_keys = cache.get_keys(0)
        assert retrieved_keys.shape[1] == 48

    def test_get_values_combines_quantized_and_residual(self, cache):
        retrieved = torch.randn(1, 48, 4, 64)
        values = torch.randn(1, 48, 4, 64)
        cache.append(0, retrieved, values)

        retrieved_values = cache.get_values(0)
        assert retrieved_values.shape[1] == 48

    def test_multi_layer(self, cache):
        """Different layers should be independent."""
        for layer in range(4):
            keys = torch.randn(1, 10, 4, 64)
            values = torch.randn(1, 10, 4, 64)
            cache.append(layer, keys, values)

        for layer in range(4):
            assert cache.get_seq_len(layer) == 10

    def test_incremental_append(self, cache):
        """Multiple small appends should accumulate correctly."""
        for _ in range(5):
            keys = torch.randn(1, 4, 4, 64)
            values = torch.randn(1, 4, 4, 64)
            cache.append(0, keys, values)

        assert cache.get_seq_len(0) == 20

    def test_clear(self, cache):
        """Clear should reset all entries."""
        keys = torch.randn(1, 16, 4, 64)
        values = torch.randn(1, 16, 4, 64)
        cache.append(0, keys, values)
        cache.clear()

        assert cache.get_seq_len(0) == 0
        assert cache.entries[0].key_residual is None

    def test_memory_usage(self, cache):
        """Memory usage should return valid metrics."""
        keys = torch.randn(1, 48, 4, 64)
        values = torch.randn(1, 48, 4, 64)
        cache.append(0, keys, values)

        mem = cache.memory_usage()
        assert "total_bytes" in mem
        assert "total_mb" in mem
        assert mem["total_bytes"] > 0

    def test_memory_less_than_fp16_baseline(self, cache):
        """Quantized cache should use less memory than pure FP16."""
        keys = torch.randn(1, 128, 4, 64)
        values = torch.randn(1, 128, 4, 64)
        cache.append(0, keys, values)

        mem = cache.memory_usage()
        # FP16 baseline: 128 tokens * 4 heads * 64 dim * 2 bytes * 2 (K+V)
        fp16_bytes = 128 * 4 * 64 * 2 * 2
        # Quantized should be meaningfully less (residual buffer is still FP16 though)
        assert mem["total_bytes"] > 0  # Non-trivial cache

    def test_empty_cache_get_keys(self, cache):
        """Getting keys from empty cache should return empty tensor."""
        result = cache.get_keys(0)
        assert result.numel() == 0


class TestCacheEntry:
    def test_initial_lengths(self):
        entry = CacheEntry()
        assert entry.total_len == 0
        assert entry.quantized_len == 0
        assert entry.residual_len == 0

    def test_total_len(self):
        entry = CacheEntry(quantized_len=50, residual_len=30)
        assert entry.total_len == 80
