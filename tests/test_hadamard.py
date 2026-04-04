"""Tests for Hadamard rotation transforms."""

import pytest
import torch
from turboquant.core.hadamard import HadamardTransform, pad_to_power_of_2, unpad


class TestHadamardTransform:
    """Test the Fast Walsh-Hadamard Transform implementation."""

    def test_dimension_validation(self):
        """Reject non-power-of-2 dimensions."""
        with pytest.raises(ValueError, match="power of 2"):
            HadamardTransform(dim=5)
        with pytest.raises(ValueError, match="power of 2"):
            HadamardTransform(dim=0)
        # Valid dimensions should work
        for dim in [1, 2, 4, 8, 16, 32, 64, 128]:
            HadamardTransform(dim=dim)

    def test_output_shape(self):
        """Output shape matches input shape."""
        ht = HadamardTransform(dim=64)
        x = torch.randn(10, 64)
        y = ht.forward(x)
        assert y.shape == x.shape

    def test_output_shape_batched(self):
        """Works with arbitrary batch dimensions."""
        ht = HadamardTransform(dim=32)
        for shape in [(32,), (4, 32), (2, 3, 32), (2, 3, 4, 32)]:
            x = torch.randn(*shape)
            y = ht.forward(x)
            assert y.shape == shape

    def test_orthogonality_preserves_norm(self):
        """Hadamard transform preserves vector norms (orthogonal property)."""
        ht = HadamardTransform(dim=128)
        x = torch.randn(100, 128)
        y = ht.forward(x)

        orig_norms = torch.norm(x, dim=-1)
        transformed_norms = torch.norm(y, dim=-1)
        torch.testing.assert_close(orig_norms, transformed_norms, atol=1e-4, rtol=1e-4)

    def test_preserves_dot_products(self):
        """Orthogonal transform preserves inner products between vectors."""
        ht = HadamardTransform(dim=64)
        a = torch.randn(64)
        b = torch.randn(64)

        dot_original = torch.dot(a, b)
        dot_transformed = torch.dot(ht.forward(a.unsqueeze(0)).squeeze(),
                                     ht.forward(b.unsqueeze(0)).squeeze())
        torch.testing.assert_close(dot_original, dot_transformed, atol=1e-4, rtol=1e-4)

    def test_invertibility(self):
        """Forward + inverse should recover the original vector."""
        ht = HadamardTransform(dim=128, seed=99)
        x = torch.randn(50, 128)

        y = ht.forward(x)
        x_reconstructed = ht.inverse(y)
        torch.testing.assert_close(x, x_reconstructed, atol=1e-4, rtol=1e-4)

    def test_different_seeds_give_different_transforms(self):
        """Different random seeds produce different rotations."""
        ht1 = HadamardTransform(dim=32, seed=1)
        ht2 = HadamardTransform(dim=32, seed=2)
        x = torch.randn(10, 32)

        y1 = ht1.forward(x)
        y2 = ht2.forward(x)
        assert not torch.allclose(y1, y2)

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical transforms."""
        ht1 = HadamardTransform(dim=64, seed=42)
        ht2 = HadamardTransform(dim=64, seed=42)
        x = torch.randn(10, 64)

        y1 = ht1.forward(x)
        y2 = ht2.forward(x)
        torch.testing.assert_close(y1, y2)

    def test_post_rotation_distribution(self):
        """After rotation, unit vector coordinates should be approximately Gaussian."""
        dim = 128
        ht = HadamardTransform(dim=dim)

        # Create random unit vectors
        x = torch.randn(10000, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        y = ht.forward(x)

        # Post-rotation: each coordinate should have std ~ 1/sqrt(dim)
        expected_std = 1.0 / (dim ** 0.5)
        actual_std = y.std(dim=0).mean().item()
        assert abs(actual_std - expected_std) < 0.02, (
            f"Expected std ~{expected_std:.4f}, got {actual_std:.4f}"
        )

    def test_dimension_mismatch_raises(self):
        """Forward should fail if input dim doesn't match."""
        ht = HadamardTransform(dim=32)
        x = torch.randn(10, 64)
        with pytest.raises(ValueError, match="Expected last dim"):
            ht.forward(x)


class TestPadding:
    def test_pad_power_of_2(self):
        x = torch.randn(5, 100)
        padded, new_dim = pad_to_power_of_2(x)
        assert new_dim == 128
        assert padded.shape == (5, 128)
        # Original values preserved
        torch.testing.assert_close(padded[:, :100], x)

    def test_no_pad_if_already_power_of_2(self):
        x = torch.randn(5, 64)
        padded, new_dim = pad_to_power_of_2(x)
        assert new_dim == 64
        torch.testing.assert_close(padded, x)

    def test_unpad(self):
        x = torch.randn(5, 128)
        unpadded = unpad(x, 100)
        assert unpadded.shape == (5, 100)
        torch.testing.assert_close(unpadded, x[:, :100])
