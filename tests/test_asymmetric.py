"""Tests for KIVI-style asymmetric quantization."""

import pytest
import torch
from turboquant.quant.asymmetric import AsymmetricQuantizer


class TestAsymmetricQuantizer:
    """Test per-channel key / per-token value quantization."""

    @pytest.fixture
    def aq(self):
        return AsymmetricQuantizer(key_bits=4, value_bits=2)

    @pytest.fixture
    def sample_kv(self):
        """Simulated KV cache: (batch=2, seq=16, heads=8, dim=64)."""
        torch.manual_seed(42)
        keys = torch.randn(2, 16, 8, 64)
        values = torch.randn(2, 16, 8, 64)
        # Add outlier channels to keys (realistic pattern)
        keys[:, :, :, 0] *= 10  # Channel 0 is an outlier
        keys[:, :, :, 32] *= 8  # Channel 32 is an outlier
        return keys, values

    def test_key_quantize_dequantize_shape(self, aq, sample_kv):
        """Quantized keys should maintain spatial dimensions."""
        keys, _ = sample_kv
        q_keys, meta = aq.quantize_keys(keys)
        assert q_keys.shape == keys.shape
        assert meta["scales"] is not None
        assert meta["zeros"] is not None

    def test_value_quantize_dequantize_shape(self, aq, sample_kv):
        _, values = sample_kv
        q_vals, meta = aq.quantize_values(values)
        assert q_vals.shape == values.shape

    def test_key_per_channel_scales_shape(self, aq, sample_kv):
        """Per-channel: scales should have seq dim collapsed to 1."""
        keys, _ = sample_kv
        _, meta = aq.quantize_keys(keys)
        # (batch, 1, heads, dim) for per-channel
        assert meta["scales"].shape[1] == 1
        assert meta["scales"].shape[-1] == keys.shape[-1]

    def test_value_per_token_scales_shape(self, aq, sample_kv):
        """Per-token: scales should have dim collapsed to 1."""
        _, values = sample_kv
        _, meta = aq.quantize_values(values)
        # (batch, seq, heads, 1) for per-token
        assert meta["scales"].shape[-1] == 1
        assert meta["scales"].shape[1] == values.shape[1]

    def test_key_roundtrip_quality(self, aq, sample_kv):
        """4-bit per-channel key quantization should have high cosine similarity."""
        keys, _ = sample_kv
        q_keys, meta = aq.quantize_keys(keys)
        recon = aq.dequantize_keys(q_keys, meta)

        cos_sim = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, 64), recon.reshape(-1, 64), dim=-1
        )
        assert cos_sim.mean() > 0.95, f"Key cosine sim {cos_sim.mean():.4f} too low"

    def test_value_roundtrip_quality(self, aq, sample_kv):
        """2-bit per-token value quantization should still be reasonable."""
        _, values = sample_kv
        q_vals, meta = aq.quantize_values(values)
        recon = aq.dequantize_values(q_vals, meta)

        cos_sim = torch.nn.functional.cosine_similarity(
            values.reshape(-1, 64), recon.reshape(-1, 64), dim=-1
        )
        # 2-bit is aggressive; cosine sim > 0.85 is acceptable
        assert cos_sim.mean() > 0.85, f"Value cosine sim {cos_sim.mean():.4f} too low"

    def test_per_channel_better_for_keys_with_outliers(self, sample_kv):
        """Per-channel should outperform per-token for keys with outlier channels."""
        keys, _ = sample_kv

        aq_channel = AsymmetricQuantizer(key_bits=4, value_bits=4, key_mode="per_channel")
        aq_token = AsymmetricQuantizer(key_bits=4, value_bits=4, key_mode="per_token")

        q_ch, meta_ch = aq_channel.quantize_keys(keys)
        recon_ch = aq_channel.dequantize_keys(q_ch, meta_ch)

        q_tok, meta_tok = aq_token._quantize_per_token(keys, 4, 15)
        recon_tok = keys.float() * 0  # need to dequantize
        recon_tok = q_tok.float() * meta_tok["scales"] + meta_tok["zeros"]

        mse_channel = ((keys - recon_ch) ** 2).mean().item()
        mse_token = ((keys - recon_tok) ** 2).mean().item()

        assert mse_channel < mse_token, (
            f"Per-channel MSE {mse_channel:.6f} should be < per-token MSE {mse_token:.6f}"
        )

    def test_quantized_values_in_range(self, aq, sample_kv):
        """Quantized indices should be in [0, 2^bits - 1]."""
        keys, values = sample_kv
        q_keys, _ = aq.quantize_keys(keys)
        q_vals, _ = aq.quantize_values(values)

        assert q_keys.min() >= 0
        assert q_keys.max() <= 15  # 4-bit
        assert q_vals.min() >= 0
        assert q_vals.max() <= 3   # 2-bit

    def test_compute_error_metrics(self, aq, sample_kv):
        """Error metrics should be computed correctly."""
        keys, _ = sample_kv
        q_keys, meta = aq.quantize_keys(keys)
        error = aq.compute_error(keys, q_keys, meta)

        assert "mse" in error
        assert "relative_error" in error
        assert "cosine_similarity" in error
        assert error["mse"] >= 0
        assert 0 <= error["cosine_similarity"] <= 1

    def test_asymmetric_4bit_key_2bit_value(self):
        """The recommended 4K+2V config should work end-to-end."""
        aq = AsymmetricQuantizer(key_bits=4, value_bits=2)
        torch.manual_seed(42)
        keys = torch.randn(1, 32, 8, 128)
        values = torch.randn(1, 32, 8, 128)

        q_k, meta_k = aq.quantize_keys(keys)
        q_v, meta_v = aq.quantize_values(values)
        recon_k = aq.dequantize_keys(q_k, meta_k)
        recon_v = aq.dequantize_values(q_v, meta_v)

        assert recon_k.shape == keys.shape
        assert recon_v.shape == values.shape
