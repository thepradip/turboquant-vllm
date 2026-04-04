"""Tests for PolarQuant vector compression."""

import pytest
import torch
from turboquant.core.polar_quant import PolarQuant, PolarQuantOutput


class TestPolarQuant:
    """Test PolarQuant magnitude/direction decomposition and quantization."""

    @pytest.fixture
    def pq(self):
        return PolarQuant(dim=128, bits=4)

    @pytest.fixture
    def sample_vectors(self):
        torch.manual_seed(42)
        return torch.randn(10, 128)

    def test_encode_output_types(self, pq, sample_vectors):
        """Encode should return proper PolarQuantOutput."""
        output = pq.encode(sample_vectors)
        assert isinstance(output, PolarQuantOutput)
        assert output.magnitudes.dtype == torch.float16
        assert output.quantized_indices.dtype in (torch.int16, torch.int32, torch.int64)

    def test_encode_shapes(self, pq, sample_vectors):
        """Output shapes should match input batch dimensions."""
        output = pq.encode(sample_vectors)
        assert output.magnitudes.shape == (10,)

    def test_decode_shape_matches_input(self, pq, sample_vectors):
        """Decoded vectors should have same shape as input."""
        output = pq.encode(sample_vectors)
        decoded = pq.decode(output)
        assert decoded.shape == sample_vectors.shape

    def test_magnitude_preservation(self, pq, sample_vectors):
        """Magnitudes should be close to original vector norms."""
        output = pq.encode(sample_vectors)
        original_norms = torch.norm(sample_vectors, dim=-1)
        encoded_magnitudes = output.magnitudes.float()
        # FP16 magnitudes should be within ~0.1% of originals
        rel_error = (original_norms - encoded_magnitudes).abs() / original_norms
        assert rel_error.max() < 0.01

    def test_direction_is_unit_vector(self, pq, sample_vectors):
        """After encode+decode, direction component should be unit-normalized."""
        output = pq.encode(sample_vectors)
        decoded = pq.decode(output)
        norms = torch.norm(decoded, dim=-1)
        expected_norms = output.magnitudes.float()
        # Norms of decoded vectors should match stored magnitudes
        rel_error = (norms - expected_norms).abs() / expected_norms.clamp(min=1e-6)
        assert rel_error.max() < 0.15  # Allow some quantization error

    def test_high_cosine_similarity_4bit(self, pq, sample_vectors):
        """4-bit PolarQuant should achieve >0.95 cosine similarity."""
        output = pq.encode(sample_vectors)
        decoded = pq.decode(output)
        metrics = pq.compute_quality_metrics(sample_vectors, decoded)
        assert metrics["cosine_similarity_mean"] > 0.95, (
            f"Mean cosine similarity {metrics['cosine_similarity_mean']:.4f} < 0.95"
        )

    def test_quality_improves_with_bits(self):
        """Higher bit-width should give better reconstruction."""
        torch.manual_seed(42)
        x = torch.randn(100, 128)
        cosine_sims = []

        for bits in [2, 3, 4]:
            pq = PolarQuant(dim=128, bits=bits)
            output = pq.encode(x)
            decoded = pq.decode(output)
            metrics = pq.compute_quality_metrics(x, decoded)
            cosine_sims.append(metrics["cosine_similarity_mean"])

        for i in range(len(cosine_sims) - 1):
            assert cosine_sims[i] < cosine_sims[i + 1], (
                f"Expected quality to improve: {cosine_sims}"
            )

    def test_non_power_of_2_dim(self):
        """Should handle dimensions that aren't powers of 2 via padding."""
        pq = PolarQuant(dim=100, bits=4)
        x = torch.randn(5, 100)
        output = pq.encode(x)
        decoded = pq.decode(output)
        assert decoded.shape == (5, 100)

    def test_zero_vector_handling(self, pq):
        """Should handle zero vectors without NaN/Inf."""
        x = torch.zeros(3, 128)
        output = pq.encode(x)
        decoded = pq.decode(output)
        assert not torch.isnan(decoded).any()
        assert not torch.isinf(decoded).any()

    def test_large_magnitude_vectors(self, pq):
        """Should handle very large vectors."""
        x = torch.randn(5, 128) * 1000
        output = pq.encode(x)
        decoded = pq.decode(output)
        assert not torch.isnan(decoded).any()

    def test_pack_unpack_roundtrip(self, pq, sample_vectors):
        """Packed encode/decode should match unpacked."""
        output_unpacked = pq.encode(sample_vectors, pack=False)
        output_packed = pq.encode(sample_vectors, pack=True)

        decoded_unpacked = pq.decode(output_unpacked)
        decoded_packed = pq.decode(output_packed)

        torch.testing.assert_close(decoded_unpacked, decoded_packed, atol=1e-4, rtol=1e-4)

    def test_batched_encode_decode(self):
        """Should work with multi-dimensional batch inputs."""
        pq = PolarQuant(dim=64, bits=4)
        x = torch.randn(2, 3, 4, 64)
        output = pq.encode(x)
        decoded = pq.decode(output)
        assert decoded.shape == x.shape

    def test_quality_metrics_keys(self, pq, sample_vectors):
        """compute_quality_metrics should return all expected keys."""
        output = pq.encode(sample_vectors)
        decoded = pq.decode(output)
        metrics = pq.compute_quality_metrics(sample_vectors, decoded)
        expected_keys = [
            "cosine_similarity_mean", "cosine_similarity_min",
            "cosine_similarity_std", "relative_error_mean",
            "relative_error_max", "mse",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
