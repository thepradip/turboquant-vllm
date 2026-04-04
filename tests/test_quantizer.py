"""Tests for the main TurboQuantizer orchestrator."""

import pytest
import torch
from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer


class TestTurboQuantizer:
    """Test the full quantization pipeline orchestration."""

    @pytest.fixture
    def config_4bit(self):
        return TurboQuantConfig.turbo_4bit(
            num_heads=8, num_kv_heads=4, head_dim=64
        )

    @pytest.fixture
    def config_kivi(self):
        return TurboQuantConfig.kivi_2bit(
            num_heads=8, num_kv_heads=4, head_dim=64
        )

    @pytest.fixture
    def config_1bit(self):
        return TurboQuantConfig.onebit_extreme(
            num_heads=8, num_kv_heads=4, head_dim=128
        )

    @pytest.fixture
    def sample_kv(self):
        """(batch=2, seq=8, num_kv_heads=4, head_dim=64)."""
        torch.manual_seed(42)
        keys = torch.randn(2, 8, 4, 64)
        values = torch.randn(2, 8, 4, 64)
        return keys, values

    @pytest.fixture
    def sample_kv_128(self):
        """Head dim 128 for 1-bit tests."""
        torch.manual_seed(42)
        return torch.randn(2, 8, 4, 128), torch.randn(2, 8, 4, 128)

    # -- TurboQuant 4-bit tests --

    def test_4bit_encode_decode_keys(self, config_4bit, sample_kv):
        """4-bit key encode/decode roundtrip."""
        quantizer = TurboQuantizer(config_4bit)
        keys, _ = sample_kv
        q_keys, meta = quantizer.encode_keys(keys)
        recon = quantizer.decode_keys(q_keys, meta)
        assert recon.shape == keys.shape

    def test_4bit_encode_decode_values(self, config_4bit, sample_kv):
        """4-bit value encode/decode roundtrip."""
        quantizer = TurboQuantizer(config_4bit)
        _, values = sample_kv
        q_vals, meta = quantizer.encode_values(values)
        recon = quantizer.decode_values(q_vals, meta)
        assert recon.shape == values.shape

    def test_4bit_key_quality(self, config_4bit, sample_kv):
        """4-bit keys should have high cosine similarity."""
        quantizer = TurboQuantizer(config_4bit)
        keys, _ = sample_kv
        q_keys, meta = quantizer.encode_keys(keys)
        recon = quantizer.decode_keys(q_keys, meta)

        cos_sim = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, 64), recon.reshape(-1, 64), dim=-1
        )
        assert cos_sim.mean() > 0.90, f"4-bit key cosine sim: {cos_sim.mean():.4f}"

    def test_4bit_value_quality(self, config_4bit, sample_kv):
        """4-bit values should have high cosine similarity."""
        quantizer = TurboQuantizer(config_4bit)
        _, values = sample_kv
        q_vals, meta = quantizer.encode_values(values)
        recon = quantizer.decode_values(q_vals, meta)

        cos_sim = torch.nn.functional.cosine_similarity(
            values.reshape(-1, 64), recon.reshape(-1, 64), dim=-1
        )
        assert cos_sim.mean() > 0.90, f"4-bit value cosine sim: {cos_sim.mean():.4f}"

    # -- KIVI 2-bit tests --

    def test_kivi_encode_decode_keys(self, config_kivi, sample_kv):
        quantizer = TurboQuantizer(config_kivi)
        keys, _ = sample_kv
        q_keys, meta = quantizer.encode_keys(keys)
        recon = quantizer.decode_keys(q_keys, meta)
        assert recon.shape == keys.shape

    def test_kivi_encode_decode_values(self, config_kivi, sample_kv):
        quantizer = TurboQuantizer(config_kivi)
        _, values = sample_kv
        q_vals, meta = quantizer.encode_values(values)
        recon = quantizer.decode_values(q_vals, meta)
        assert recon.shape == values.shape

    def test_kivi_asymmetric_bits(self, config_kivi):
        """KIVI should use different bits for keys vs values."""
        assert config_kivi.effective_key_bits == 4
        assert config_kivi.effective_value_bits == 2

    # -- 1-bit tests --

    def test_1bit_encode_decode(self, config_1bit, sample_kv_128):
        quantizer = TurboQuantizer(config_1bit)
        keys, values = sample_kv_128
        q_keys, meta_k = quantizer.encode_keys(keys)
        q_vals, meta_v = quantizer.encode_values(values)
        recon_k = quantizer.decode_keys(q_keys, meta_k)
        recon_v = quantizer.decode_values(q_vals, meta_v)
        assert recon_k.shape == keys.shape
        assert recon_v.shape == values.shape

    # -- Compression ratio tests --

    def test_compression_ratio_4bit(self, config_4bit):
        quantizer = TurboQuantizer(config_4bit)
        info = quantizer.compute_compression_ratio(1024 * 1024)
        assert info["compression_ratio"] > 3.5
        assert info["memory_savings_pct"] > 70

    def test_compression_ratio_1bit(self, config_1bit):
        quantizer = TurboQuantizer(config_1bit)
        info = quantizer.compute_compression_ratio(1024 * 1024)
        assert info["compression_ratio"] > 14
        assert info["memory_savings_pct"] > 90

    # -- Config presets --

    def test_turbo_4bit_preset(self):
        config = TurboQuantConfig.turbo_4bit()
        assert config.enable_polar_quant is True
        assert config.enable_qjl is False
        assert config.enable_hadamard is True
        assert config.kv_bits == 4

    def test_kivi_2bit_preset(self):
        config = TurboQuantConfig.kivi_2bit()
        assert config.enable_polar_quant is False
        assert config.enable_asymmetric is True
        assert config.kv_bits == 2

    def test_turbo_3bit_preset(self):
        config = TurboQuantConfig.turbo_3bit()
        assert config.enable_polar_quant is True
        assert config.enable_qjl is True
        assert config.kv_bits == 3

    def test_onebit_preset(self):
        config = TurboQuantConfig.onebit_extreme()
        assert config.enable_onebit is True
        assert config.kv_bits == 1

    # -- Config YAML --

    def test_config_yaml_roundtrip(self, tmp_path, config_4bit):
        path = tmp_path / "test_config.yaml"
        config_4bit.to_yaml(path)
        loaded = TurboQuantConfig.from_yaml(path)
        assert loaded.kv_bits == config_4bit.kv_bits
        assert loaded.enable_polar_quant == config_4bit.enable_polar_quant
        assert loaded.head_dim == config_4bit.head_dim

    # -- TurboQuant 3-bit with QJL tests --

    def test_3bit_qjl_encode_decode_keys(self):
        """3-bit with QJL should encode/decode keys with quality."""
        config = TurboQuantConfig.turbo_3bit(
            num_heads=8, num_kv_heads=4, head_dim=64
        )
        quantizer = TurboQuantizer(config)
        torch.manual_seed(42)
        keys = torch.randn(2, 8, 4, 64)
        q_keys, meta = quantizer.encode_keys(keys)
        assert "qjl_signs" in meta
        recon = quantizer.decode_keys(q_keys, meta)
        assert recon.shape == keys.shape
        assert not torch.isnan(recon).any()

    def test_3bit_qjl_encode_decode_values(self):
        """3-bit with QJL should encode/decode values."""
        config = TurboQuantConfig.turbo_3bit(
            num_heads=8, num_kv_heads=4, head_dim=64
        )
        quantizer = TurboQuantizer(config)
        torch.manual_seed(42)
        values = torch.randn(2, 8, 4, 64)
        q_vals, meta = quantizer.encode_values(values)
        recon = quantizer.decode_values(q_vals, meta)
        assert recon.shape == values.shape

    def test_3bit_qjl_quality(self):
        """3-bit QJL-corrected should maintain reasonable quality."""
        config = TurboQuantConfig.turbo_3bit(
            num_heads=8, num_kv_heads=4, head_dim=64
        )
        quantizer = TurboQuantizer(config)
        torch.manual_seed(42)
        keys = torch.randn(1, 32, 4, 64)
        q_keys, meta = quantizer.encode_keys(keys)
        recon = quantizer.decode_keys(q_keys, meta)
        cos_sim = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, 64), recon.reshape(-1, 64), dim=-1
        )
        assert cos_sim.mean() > 0.85, f"3-bit QJL cosine sim: {cos_sim.mean():.4f}"

    # -- Device transfer tests --

    def test_to_device_cpu(self, config_4bit):
        """Quantizer.to() should move components to target device."""
        quantizer = TurboQuantizer(config_4bit)
        quantizer = quantizer.to("cpu")
        assert quantizer.device == "cpu"
        # Should still work after device transfer
        torch.manual_seed(42)
        keys = torch.randn(1, 4, 4, 64)
        q, meta = quantizer.encode_keys(keys)
        recon = quantizer.decode_keys(q, meta)
        assert recon.shape == keys.shape

    # -- No NaN/Inf tests --

    def test_no_nan_inf_4bit(self, config_4bit):
        quantizer = TurboQuantizer(config_4bit)
        x = torch.randn(2, 4, 4, 64)
        q, meta = quantizer.encode_keys(x)
        recon = quantizer.decode_keys(q, meta)
        assert not torch.isnan(recon).any()
        assert not torch.isinf(recon).any()

    def test_no_nan_inf_1bit(self, config_1bit):
        quantizer = TurboQuantizer(config_1bit)
        x = torch.randn(2, 4, 4, 128)
        q, meta = quantizer.encode_keys(x)
        recon = quantizer.decode_keys(q, meta)
        assert not torch.isnan(recon).any()
        assert not torch.isinf(recon).any()
