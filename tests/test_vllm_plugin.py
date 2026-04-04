"""Tests for vLLM plugin integration."""

import pytest
import torch
from turboquant.vllm_plugin import TurboQuantKVHook


class TestTurboQuantKVHook:
    """Test the standalone KV hook (works without vLLM installed)."""

    @pytest.fixture
    def hook(self):
        return TurboQuantKVHook(
            num_heads=32, num_kv_heads=8, head_dim=128,
            kv_bits=4, device="cpu",
        )

    def test_encode_decode_keys_shape(self, hook):
        """vLLM format: (batch, heads, seq, dim)."""
        keys = torch.randn(1, 8, 64, 128)
        encoded = hook.encode_keys(keys, layer_idx=0)
        decoded = hook.decode_keys(encoded, layer_idx=0)
        assert decoded.shape == keys.shape

    def test_encode_decode_values_shape(self, hook):
        values = torch.randn(1, 8, 64, 128)
        encoded = hook.encode_values(values, layer_idx=0)
        decoded = hook.decode_values(encoded, layer_idx=0)
        assert decoded.shape == values.shape

    def test_quality_preserved(self, hook):
        torch.manual_seed(42)
        keys = torch.randn(1, 8, 256, 128)
        encoded = hook.encode_keys(keys, layer_idx=0)
        decoded = hook.decode_keys(encoded, layer_idx=0)

        cos_sim = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, 128), decoded.reshape(-1, 128), dim=-1
        )
        assert cos_sim.mean() > 0.99, f"Quality too low: {cos_sim.mean():.4f}"

    def test_multi_layer(self, hook):
        """Different layers should be independent."""
        k0 = torch.randn(1, 8, 32, 128)
        k1 = torch.randn(1, 8, 32, 128)

        hook.encode_keys(k0, layer_idx=0)
        hook.encode_keys(k1, layer_idx=1)

        d0 = hook.decode_keys(hook.encode_keys(k0, layer_idx=0), layer_idx=0)
        d1 = hook.decode_keys(hook.encode_keys(k1, layer_idx=1), layer_idx=1)

        assert d0.shape == k0.shape
        assert d1.shape == k1.shape

    def test_no_nan_inf(self, hook):
        keys = torch.randn(1, 8, 128, 128)
        encoded = hook.encode_keys(keys)
        decoded = hook.decode_keys(encoded)
        assert not torch.isnan(decoded).any()
        assert not torch.isinf(decoded).any()

    def test_vllm_format_batch(self, hook):
        """Batch size > 1."""
        keys = torch.randn(4, 8, 64, 128)
        encoded = hook.encode_keys(keys)
        decoded = hook.decode_keys(encoded)
        assert decoded.shape == (4, 8, 64, 128)


class TestPluginRegistration:
    """Test that registration function is safe to call."""

    def test_register_idempotent(self):
        from turboquant.vllm_plugin import register
        register()
        register()  # Should not error on second call

    def test_register_without_vllm(self):
        """Should not crash when vLLM is not installed."""
        from turboquant.vllm_plugin import register
        register()  # Graceful fallback
