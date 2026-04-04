"""Tests for 1-bit Bonsai-inspired quantization."""

import pytest
import torch
from turboquant.quant.onebit import OneBitQuantizer


class TestOneBitQuantizer:
    """Test Q1_0_g128 format quantization."""

    @pytest.fixture
    def ob(self):
        return OneBitQuantizer(group_size=128)

    def test_effective_bits(self, ob):
        """Q1_0_g128 should have 1.125 effective bits."""
        assert abs(ob.effective_bits - 1.125) < 1e-6

    def test_effective_bits_different_group_sizes(self):
        for gs, expected in [(64, 1.25), (128, 1.125), (256, 1.0625)]:
            ob = OneBitQuantizer(group_size=gs)
            assert abs(ob.effective_bits - expected) < 1e-6

    def test_quantize_output_shapes(self, ob):
        """Packed signs and scales should have correct shapes."""
        x = torch.randn(10, 128)
        packed, scales = ob.quantize(x)

        # scales: (10, 1) -- one scale per 128-element group, 128/128 = 1 group
        assert scales.shape == (10, 1)
        assert scales.dtype == torch.float16

        # packed: (10, 16) -- 128 bits / 8 = 16 bytes per vector
        assert packed.shape == (10, 16)
        assert packed.dtype == torch.uint8

    def test_dequantize_shape(self, ob):
        """Dequantized output should match original shape."""
        x = torch.randn(5, 128)
        packed, scales = ob.quantize(x)
        recon = ob.dequantize(packed, scales, 128)
        assert recon.shape == (5, 128)

    def test_sign_preservation(self, ob):
        """Signs of reconstructed values should mostly match original signs."""
        torch.manual_seed(42)
        x = torch.randn(100, 128) * 2  # Scale up for clear signs

        packed, scales = ob.quantize(x)
        recon = ob.dequantize(packed, scales, 128)

        # Check sign agreement
        orig_signs = (x >= 0).float()
        recon_signs = (recon >= 0).float()
        agreement = (orig_signs == recon_signs).float().mean()
        assert agreement > 0.99, f"Sign agreement {agreement:.2%} too low"

    def test_scale_represents_mean_abs(self, ob):
        """Scales should approximate mean absolute value of each group."""
        torch.manual_seed(42)
        x = torch.randn(10, 128)
        _, scales = ob.quantize(x)

        expected_scales = x.abs().mean(dim=-1)
        actual_scales = scales.float().squeeze(-1)
        # FP16 precision
        rel_error = (expected_scales - actual_scales).abs() / expected_scales.clamp(min=1e-6)
        assert rel_error.max() < 0.01

    def test_compression_ratio(self, ob):
        """1-bit should achieve ~14x compression vs FP16."""
        ratio = 16.0 / ob.effective_bits
        assert ratio > 14.0

    def test_non_divisible_dim(self):
        """Should handle dimensions not divisible by group_size."""
        ob = OneBitQuantizer(group_size=128)
        x = torch.randn(5, 100)  # 100 is not divisible by 128
        packed, scales = ob.quantize(x)
        recon = ob.dequantize(packed, scales, 100)
        assert recon.shape == (5, 100)

    def test_quality_metrics(self, ob):
        """compute_quality should return comprehensive metrics."""
        x = torch.randn(50, 128)
        metrics = ob.compute_quality(x)

        assert "mse" in metrics
        assert "cosine_similarity_mean" in metrics
        assert "snr_db" in metrics
        assert "effective_bits" in metrics
        assert "compression_ratio" in metrics
        assert "memory_reduction_pct" in metrics
        assert metrics["memory_reduction_pct"] > 90  # >90% memory reduction

    def test_cosine_similarity_reasonable(self, ob):
        """1-bit should achieve reasonable (not perfect) cosine similarity."""
        torch.manual_seed(42)
        x = torch.randn(100, 128)
        metrics = ob.compute_quality(x)
        # 1-bit is aggressive; cosine sim > 0.5 is expected for random vectors
        assert metrics["cosine_similarity_mean"] > 0.5

    def test_large_group_size(self):
        ob = OneBitQuantizer(group_size=256)
        x = torch.randn(5, 256)
        packed, scales = ob.quantize(x)
        recon = ob.dequantize(packed, scales, 256)
        assert recon.shape == x.shape

    def test_invalid_group_size(self):
        with pytest.raises(ValueError):
            OneBitQuantizer(group_size=0)

    def test_bit_packing_correctness(self, ob):
        """Verify bit packing/unpacking preserves all bits."""
        # Create known bit pattern
        x = torch.randn(1, 128)
        packed, scales = ob.quantize(x)
        recon = ob.dequantize(packed, scales, 128)

        # Re-quantize and compare packed data
        packed2, scales2 = ob.quantize(recon)
        # The packed bits should be identical (signs preserved)
        assert torch.equal(packed, packed2)
