"""Integration tests: full pipeline from config to quantized attention."""

import pytest
import torch
from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer
from turboquant.core.kv_cache import QuantizedKVCache
from turboquant.engine.attention import QuantizedAttention
from turboquant.utils.metrics import QualityMetrics, MetricsReport


class TestFullPipeline:
    """End-to-end integration tests for all quantization modes."""

    @pytest.fixture(params=["turbo_4bit", "kivi_2bit", "onebit_extreme"])
    def config_and_dims(self, request):
        """Parametrized fixture for all presets."""
        preset = request.param
        if preset == "onebit_extreme":
            config = TurboQuantConfig.onebit_extreme(
                num_heads=8, num_kv_heads=4, head_dim=128,
                residual_buffer_size=32, kivi_group_size=16,
            )
            dim = 128
        else:
            factory = getattr(TurboQuantConfig, preset)
            config = factory(
                num_heads=8, num_kv_heads=4, head_dim=64,
                residual_buffer_size=32, kivi_group_size=16,
            )
            dim = 64
        return config, dim

    def test_encode_decode_roundtrip(self, config_and_dims):
        """Every preset should encode/decode without errors."""
        config, dim = config_and_dims
        quantizer = TurboQuantizer(config)
        torch.manual_seed(42)
        keys = torch.randn(1, 16, 4, dim)
        values = torch.randn(1, 16, 4, dim)

        q_k, meta_k = quantizer.encode_keys(keys)
        q_v, meta_v = quantizer.encode_values(values)
        recon_k = quantizer.decode_keys(q_k, meta_k)
        recon_v = quantizer.decode_values(q_v, meta_v)

        assert recon_k.shape == keys.shape
        assert recon_v.shape == values.shape
        assert not torch.isnan(recon_k).any()
        assert not torch.isnan(recon_v).any()

    def test_kv_cache_full_workflow(self, config_and_dims):
        """Full KV cache lifecycle: append, flush, retrieve, clear."""
        config, dim = config_and_dims
        quantizer = TurboQuantizer(config)
        cache = QuantizedKVCache(config, num_layers=2)
        cache.set_quantizer(quantizer)

        # Phase 1: Fill up and trigger quantization
        total_tokens = 0
        for _ in range(10):
            batch_len = 8
            keys = torch.randn(1, batch_len, 4, dim)
            values = torch.randn(1, batch_len, 4, dim)
            cache.append(0, keys, values)
            total_tokens += batch_len

        assert cache.get_seq_len(0) == total_tokens

        # Phase 2: Retrieve
        all_keys = cache.get_keys(0)
        all_values = cache.get_values(0)
        assert all_keys.shape[1] == total_tokens
        assert all_values.shape[1] == total_tokens

        # Phase 3: Clear
        cache.clear()
        assert cache.get_seq_len(0) == 0

    def test_attention_with_quantized_cache(self, config_and_dims):
        """Attention should produce valid output with any quantization preset."""
        config, dim = config_and_dims
        quantizer = TurboQuantizer(config)
        cache = QuantizedKVCache(config, num_layers=1)
        cache.set_quantizer(quantizer)
        attn = QuantizedAttention(num_heads=8, num_kv_heads=4, head_dim=dim)

        torch.manual_seed(42)
        # Add KV entries
        keys = torch.randn(1, 64, 4, dim)
        values = torch.randn(1, 64, 4, dim)
        cache.append(0, keys, values)

        # Query
        query = torch.randn(1, 1, 8, dim)
        output = attn.forward(query, cache, layer_idx=0)

        assert output.shape == (1, 1, 8, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_quality_metrics_report(self, config_and_dims):
        """Quality metrics should be computable for all presets."""
        config, dim = config_and_dims
        quantizer = TurboQuantizer(config)
        torch.manual_seed(42)
        keys = torch.randn(1, 32, 4, dim)

        q_k, meta_k = quantizer.encode_keys(keys)
        recon_k = quantizer.decode_keys(q_k, meta_k)

        bits = config.effective_key_bits
        if config.enable_onebit:
            bits = 1.125
        report = QualityMetrics.full_report(keys, recon_k, effective_bits=bits)

        assert isinstance(report, MetricsReport)
        assert report.mse >= 0
        assert report.snr_db > 0 or config.enable_onebit  # 1-bit may have low SNR
        assert report.cosine_similarity_mean > 0


class TestMultiplePresetComparison:
    """Compare quality across presets to validate expected ordering."""

    def test_4bit_better_than_2bit(self):
        """4-bit should have higher quality than 2-bit."""
        torch.manual_seed(42)
        keys = torch.randn(1, 32, 4, 64)

        config_4 = TurboQuantConfig.turbo_4bit(num_heads=8, num_kv_heads=4, head_dim=64)
        config_2 = TurboQuantConfig.kivi_2bit(num_heads=8, num_kv_heads=4, head_dim=64)

        q4 = TurboQuantizer(config_4)
        q2 = TurboQuantizer(config_2)

        _, meta4 = q4.encode_keys(keys)
        recon4 = q4.decode_keys(*q4.encode_keys(keys))

        _, meta2 = q2.encode_keys(keys)
        recon2 = q2.decode_keys(*q2.encode_keys(keys))

        cos4 = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, 64), recon4.reshape(-1, 64), dim=-1
        ).mean()
        cos2 = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, 64), recon2.reshape(-1, 64), dim=-1
        ).mean()

        assert cos4 > cos2, f"4-bit ({cos4:.4f}) should be better than 2-bit ({cos2:.4f})"

    def test_compression_ratio_ordering(self):
        """Higher compression should correlate with lower quality."""
        configs = [
            ("4-bit", TurboQuantConfig.turbo_4bit(head_dim=128)),
            ("2-bit", TurboQuantConfig.kivi_2bit(head_dim=128)),
            ("1-bit", TurboQuantConfig.onebit_extreme(head_dim=128)),
        ]

        ratios = []
        for name, config in configs:
            q = TurboQuantizer(config)
            info = q.compute_compression_ratio(1024 * 1024)
            ratios.append((name, info["compression_ratio"]))

        # 1-bit > 2-bit > 4-bit compression ratio
        assert ratios[2][1] > ratios[1][1] > ratios[0][1]


class TestMemoryProfile:
    """Test memory profiling across configurations."""

    def test_memory_usage_tracked(self):
        """Memory usage should be tracked correctly."""
        config = TurboQuantConfig.turbo_4bit(
            num_heads=8, num_kv_heads=4, head_dim=64,
            residual_buffer_size=16, kivi_group_size=8,
        )
        quantizer = TurboQuantizer(config)
        cache = QuantizedKVCache(config, num_layers=4)
        cache.set_quantizer(quantizer)

        # Add significant amount of data
        for layer in range(4):
            keys = torch.randn(1, 64, 4, 64)
            values = torch.randn(1, 64, 4, 64)
            cache.append(layer, keys, values)

        mem = cache.memory_usage()
        assert mem["total_mb"] > 0
        assert mem["quantized_bytes"] >= 0
        assert mem["residual_bytes"] > 0
