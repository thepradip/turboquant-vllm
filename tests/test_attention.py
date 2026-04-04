"""Tests for quantized attention computation."""

import pytest
import torch
from turboquant.engine.attention import QuantizedAttention
from turboquant.config import TurboQuantConfig
from turboquant.core.kv_cache import QuantizedKVCache
from turboquant.core.quantizer import TurboQuantizer


class TestQuantizedAttention:
    """Test attention computation with quantized KV."""

    @pytest.fixture
    def attn(self):
        return QuantizedAttention(num_heads=8, num_kv_heads=4, head_dim=64)

    def test_compute_attention_shape(self, attn):
        """Attention output should match query shape."""
        query = torch.randn(2, 4, 8, 64)   # (batch, q_len, heads, dim)
        keys = torch.randn(2, 16, 4, 64)   # (batch, kv_len, kv_heads, dim)
        values = torch.randn(2, 16, 4, 64)

        output = attn.compute_attention(query, keys, values)
        assert output.shape == (2, 4, 8, 64)

    def test_gqa_expansion(self, attn):
        """GQA should expand kv_heads to match query heads."""
        assert attn.num_groups == 2  # 8 / 4 = 2

    def test_attention_with_mask(self, attn):
        """Causal mask should work correctly."""
        query = torch.randn(1, 4, 8, 64)
        keys = torch.randn(1, 4, 4, 64)
        values = torch.randn(1, 4, 4, 64)

        # Causal mask
        mask = torch.triu(torch.ones(4, 4) * float("-inf"), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)

        output = attn.compute_attention(query, keys, values, mask)
        assert output.shape == (1, 4, 8, 64)
        assert not torch.isnan(output).any()

    def test_attention_with_kv_cache(self):
        """Test attention through the full KV cache pipeline."""
        config = TurboQuantConfig.turbo_4bit(
            num_heads=8, num_kv_heads=4, head_dim=64,
            residual_buffer_size=64,
        )
        quantizer = TurboQuantizer(config)
        cache = QuantizedKVCache(config, num_layers=1)
        cache.set_quantizer(quantizer)
        attn = QuantizedAttention(num_heads=8, num_kv_heads=4, head_dim=64)

        # Add KV entries
        keys = torch.randn(1, 16, 4, 64)
        values = torch.randn(1, 16, 4, 64)
        cache.append(0, keys, values)

        # Query
        query = torch.randn(1, 1, 8, 64)
        output = attn.forward(query, cache, layer_idx=0)
        assert output.shape == (1, 1, 8, 64)

    def test_attention_error_metrics(self, attn):
        """Error metrics should be computed between original and quantized."""
        torch.manual_seed(42)
        query = torch.randn(1, 4, 8, 64)
        orig_k = torch.randn(1, 16, 4, 64)
        orig_v = torch.randn(1, 16, 4, 64)
        # Simulate quantization noise
        quant_k = orig_k + torch.randn_like(orig_k) * 0.01
        quant_v = orig_v + torch.randn_like(orig_v) * 0.01

        metrics = attn.compute_attention_error(query, orig_k, orig_v, quant_k, quant_v)

        assert "attention_mse" in metrics
        assert "attention_cosine_similarity_mean" in metrics
        assert "attention_relative_error" in metrics
        assert metrics["attention_cosine_similarity_mean"] > 0.99  # Small noise

    def test_no_kv_heads_gqa(self):
        """Should work with num_heads == num_kv_heads (MHA)."""
        attn = QuantizedAttention(num_heads=8, num_kv_heads=8, head_dim=64)
        query = torch.randn(1, 4, 8, 64)
        keys = torch.randn(1, 16, 8, 64)
        values = torch.randn(1, 16, 8, 64)

        output = attn.compute_attention(query, keys, values)
        assert output.shape == (1, 4, 8, 64)

    def test_quantization_impact_on_attention(self):
        """4-bit quantized KV should produce attention close to FP16 baseline."""
        config = TurboQuantConfig.turbo_4bit(
            num_heads=8, num_kv_heads=4, head_dim=64,
        )
        config.enable_mixed_precision = False
        quantizer = TurboQuantizer(config)
        attn = QuantizedAttention(num_heads=8, num_kv_heads=4, head_dim=64)

        torch.manual_seed(42)
        query = torch.randn(1, 4, 8, 64)
        keys = torch.randn(1, 32, 4, 64)
        values = torch.randn(1, 32, 4, 64)

        # Quantize and dequantize
        q_k, meta_k = quantizer.encode_keys(keys)
        q_v, meta_v = quantizer.encode_values(values)
        recon_k = quantizer.decode_keys(q_k, meta_k)
        recon_v = quantizer.decode_values(q_v, meta_v)

        metrics = attn.compute_attention_error(query, keys, values, recon_k, recon_v)
        assert metrics["attention_cosine_similarity_mean"] > 0.90, (
            f"Attention cosine sim {metrics['attention_cosine_similarity_mean']:.4f} too low"
        )
