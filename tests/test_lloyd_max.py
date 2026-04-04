"""Tests for Lloyd-Max optimal scalar quantizer."""

import pytest
import torch
import math
from turboquant.core.lloyd_max import LloydMaxQuantizer


class TestLloydMaxQuantizer:
    """Test the Lloyd-Max codebook and quantization."""

    def test_codebook_size(self):
        """Codebook should have 2^bits entries."""
        for bits in [1, 2, 3, 4]:
            lm = LloydMaxQuantizer(bits=bits, dim=128)
            assert lm.codebook.shape[0] == (1 << bits)

    def test_codebook_is_sorted(self):
        """Codebook levels should be monotonically increasing."""
        lm = LloydMaxQuantizer(bits=4, dim=128)
        diffs = lm.codebook[1:] - lm.codebook[:-1]
        assert (diffs > 0).all(), "Codebook should be strictly increasing"

    def test_boundaries_between_levels(self):
        """Decision boundaries should lie between adjacent codebook levels."""
        lm = LloydMaxQuantizer(bits=4, dim=128)
        for i in range(len(lm.boundaries)):
            assert lm.boundaries[i] > lm.codebook[i]
            assert lm.boundaries[i] < lm.codebook[i + 1]

    def test_quantize_dequantize_roundtrip(self):
        """Dequantize(quantize(x)) should be close to x for values in distribution."""
        lm = LloydMaxQuantizer(bits=4, dim=128)
        std = 1.0 / math.sqrt(128)
        x = torch.randn(1000) * std

        indices = lm.quantize(x)
        reconstructed = lm.dequantize(indices)

        mse = ((x - reconstructed) ** 2).mean().item()
        # 4-bit quantization of Gaussian should have low MSE
        assert mse < std ** 2 * 0.1, f"MSE {mse} too high for 4-bit"

    def test_indices_in_range(self):
        """Quantized indices should be in [0, num_levels-1]."""
        lm = LloydMaxQuantizer(bits=4, dim=128)
        x = torch.randn(1000) * 0.1
        indices = lm.quantize(x)
        assert indices.min() >= 0
        assert indices.max() < lm.num_levels

    def test_pack_unpack_roundtrip_4bit(self):
        """Pack and unpack should preserve values for 4-bit."""
        lm = LloydMaxQuantizer(bits=4, dim=128)
        x = torch.randn(100) * 0.1
        packed, shape_info = lm.quantize_and_pack(x)

        reconstructed = lm.unpack_and_dequantize(packed, shape_info)
        # Should match direct dequantize
        indices = lm.quantize(x)
        direct_recon = lm.dequantize(indices)
        torch.testing.assert_close(reconstructed, direct_recon, atol=1e-6, rtol=1e-6)

    def test_pack_reduces_memory_4bit(self):
        """4-bit packing should use ~half the bytes of int16."""
        lm = LloydMaxQuantizer(bits=4, dim=128)
        x = torch.randn(1024) * 0.1
        packed, _ = lm.quantize_and_pack(x)

        # packed should have ceil(1024/2) = 512 uint8 elements
        assert packed.shape[0] == 512
        assert packed.dtype == torch.uint8

    def test_higher_bits_lower_mse(self):
        """More bits should give lower quantization error."""
        std = 1.0 / math.sqrt(128)
        x = torch.randn(10000) * std

        mse_values = []
        for bits in [1, 2, 3, 4]:
            lm = LloydMaxQuantizer(bits=bits, dim=128)
            mse = lm.compute_mse(x)
            mse_values.append(mse)

        # MSE should decrease with more bits
        for i in range(len(mse_values) - 1):
            assert mse_values[i] > mse_values[i + 1], (
                f"MSE at {i+1}-bit ({mse_values[i]}) should be > MSE at {i+2}-bit ({mse_values[i+1]})"
            )

    def test_different_dimensions_different_codebooks(self):
        """Different input dimensions should produce different codebooks."""
        lm64 = LloydMaxQuantizer(bits=4, dim=64)
        lm256 = LloydMaxQuantizer(bits=4, dim=256)
        assert not torch.allclose(lm64.codebook, lm256.codebook)

    def test_compute_mse_positive(self):
        """MSE should always be non-negative."""
        lm = LloydMaxQuantizer(bits=2, dim=128)
        x = torch.randn(100) * 0.1
        assert lm.compute_mse(x) >= 0

    def test_edge_case_single_element(self):
        """Should handle single element input."""
        lm = LloydMaxQuantizer(bits=4, dim=128)
        x = torch.tensor([0.05])
        indices = lm.quantize(x)
        recon = lm.dequantize(indices)
        assert recon.shape == (1,)
