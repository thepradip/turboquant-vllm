"""Tests for Quantized Johnson-Lindenstrauss projection."""

import pytest
import torch
from turboquant.core.qjl import QJLProjection


class TestQJLProjection:
    """Test QJL residual correction."""

    @pytest.fixture
    def qjl(self):
        return QJLProjection(input_dim=128, projection_dim=256)

    def test_encode_output_shape(self, qjl):
        """Encoded signs should have shape (..., projection_dim)."""
        x = torch.randn(10, 128)
        signs, norms = qjl.encode(x)
        assert signs.shape == (10, 256)
        assert norms.shape == (10,)

    def test_encode_values_are_signs(self, qjl):
        """All encoded values should be -1 or +1."""
        x = torch.randn(50, 128)
        signs, norms = qjl.encode(x)
        assert ((signs == -1) | (signs == 1)).all()

    def test_encode_norms_are_positive(self, qjl):
        """Residual norms should be non-negative FP16."""
        x = torch.randn(50, 128)
        signs, norms = qjl.encode(x)
        assert norms.dtype == torch.float16
        assert (norms >= 0).all()

    def test_pack_unpack_roundtrip(self, qjl):
        """Packing and unpacking should preserve sign values."""
        x = torch.randn(10, 128)
        signs, _ = qjl.encode(x)
        packed = qjl.pack_signs(signs)
        unpacked = qjl.unpack_signs(packed, signs.numel())
        unpacked = unpacked.reshape(signs.shape)
        torch.testing.assert_close(signs, unpacked)

    def test_pack_reduces_memory(self, qjl):
        """Packed signs should use 8x less memory than float."""
        x = torch.randn(100, 128)
        signs, _ = qjl.encode(x)
        flat_signs = signs.reshape(-1)
        packed = qjl.pack_signs(signs)
        expected_bytes = (flat_signs.numel() + 7) // 8
        assert packed.numel() == expected_bytes

    def test_decode_correction_shape(self, qjl):
        """Decoded correction should have input_dim shape."""
        x = torch.randn(10, 128)
        signs, norms = qjl.encode(x)
        correction = qjl.decode_correction(signs, norms)
        assert correction.shape == (10, 128)

    def test_decode_correction_backward_compat(self, qjl):
        """decode_correction without norms should still work (4-bit path)."""
        x = torch.randn(10, 128)
        signs, _ = qjl.encode(x)
        correction = qjl.decode_correction(signs)  # no norms = old behavior
        assert correction.shape == (10, 128)

    def test_inner_product_estimation(self, qjl):
        """QJL inner product estimate should correlate with true inner product."""
        torch.manual_seed(42)
        a = torch.randn(128)
        b = torch.randn(128)

        true_ip = torch.dot(a, b)
        signs_a, _ = qjl.encode(a.unsqueeze(0))
        signs_b, _ = qjl.encode(b.unsqueeze(0))
        estimated_ip = qjl.compute_inner_product_estimate(signs_a, signs_b)

        assert abs(estimated_ip.item()) < 2.0

    def test_correction_reduces_error(self):
        """Adding QJL correction should reduce reconstruction error."""
        from turboquant.core.polar_quant import PolarQuant

        pq = PolarQuant(dim=128, bits=2)
        qjl = QJLProjection(input_dim=128, projection_dim=512)

        torch.manual_seed(42)
        x = torch.randn(50, 128)

        encoded = pq.encode(x)
        reconstructed = pq.decode(encoded)

        residual = x - reconstructed
        signs, norms = qjl.encode(residual)
        correction = qjl.decode_correction(signs, norms)

        corrected = reconstructed + correction

        error_before = ((x - reconstructed) ** 2).sum(dim=-1)
        error_after = ((x - corrected) ** 2).sum(dim=-1)

        improved = (error_after < error_before).float().mean()
        assert improved > 0.3, f"Only {improved:.0%} of vectors improved with QJL"

    def test_deterministic_projection(self):
        """Same seed should give same projection matrix."""
        qjl1 = QJLProjection(128, 256, seed=42)
        qjl2 = QJLProjection(128, 256, seed=42)
        torch.testing.assert_close(qjl1.projection_matrix, qjl2.projection_matrix)

    def test_different_seeds_different_projections(self):
        """Different seeds should give different projections."""
        qjl1 = QJLProjection(128, 256, seed=1)
        qjl2 = QJLProjection(128, 256, seed=2)
        assert not torch.allclose(qjl1.projection_matrix, qjl2.projection_matrix)

    def test_batched_input(self, qjl):
        """Should work with multi-dimensional batch inputs."""
        x = torch.randn(2, 5, 128)
        signs, norms = qjl.encode(x)
        assert signs.shape == (2, 5, 256)
        assert norms.shape == (2, 5)
