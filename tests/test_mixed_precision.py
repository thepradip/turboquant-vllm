"""Tests for mixed-precision quantization with outlier handling."""

import pytest
import torch
from turboquant.quant.mixed_precision import MixedPrecisionQuantizer


class TestMixedPrecisionQuantizer:
    """Test outlier channel identification and mixed-precision quantization."""

    @pytest.fixture
    def mp(self):
        return MixedPrecisionQuantizer(
            outlier_fraction=0.25, outlier_bits=8, normal_bits=4
        )

    @pytest.fixture
    def data_with_outliers(self):
        """Simulated data where channels 0,1,2 have 10x magnitude."""
        torch.manual_seed(42)
        x = torch.randn(100, 32)
        x[:, :8] *= 10  # 25% outlier channels
        return x

    def test_effective_bits(self, mp):
        """Effective bits should be weighted average."""
        # 0.25 * 8 + 0.75 * 4 = 5.0
        assert abs(mp.effective_bits - 5.0) < 1e-6

    def test_identify_outliers(self, mp, data_with_outliers):
        """Should identify the high-magnitude channels as outliers."""
        outlier_idx = mp.identify_outlier_channels(data_with_outliers)
        # 25% of 32 = 8 channels
        assert len(outlier_idx) == 8
        # The top-8 magnitude channels should be in [0,7]
        for idx in outlier_idx:
            assert idx.item() < 8

    def test_extract_restore_roundtrip(self, mp, data_with_outliers):
        """Extract + restore should recover original data exactly."""
        normal, outlier_data = mp.extract_outliers(data_with_outliers)
        restored = mp.restore_outliers(
            normal, outlier_data["indices"], outlier_data["values"]
        )
        torch.testing.assert_close(restored, data_with_outliers)

    def test_normal_has_zeros_at_outlier_channels(self, mp, data_with_outliers):
        """After extraction, outlier channels in normal tensor should be zero."""
        normal, outlier_data = mp.extract_outliers(data_with_outliers)
        outlier_idx = outlier_data["indices"]
        assert (normal[:, outlier_idx] == 0).all()

    def test_outlier_int8_quantize(self, mp, data_with_outliers):
        """INT8 quantization of outlier channels."""
        _, outlier_data = mp.extract_outliers(data_with_outliers)
        values = outlier_data["values"]
        q_vals, scale, zp = mp.quantize_outliers_int8(values)

        assert q_vals.dtype == torch.uint8
        assert q_vals.min() >= 0
        assert q_vals.max() <= 255

        # Dequantize and check quality
        recon = mp.dequantize_outliers_int8(q_vals, scale, zp)
        mse = ((values - recon) ** 2).mean().item()
        assert mse < 0.1  # INT8 should be high quality

    def test_repeated_extraction_consistent(self, mp, data_with_outliers):
        """Multiple extractions should produce same results."""
        n1, o1 = mp.extract_outliers(data_with_outliers)
        n2, o2 = mp.extract_outliers(data_with_outliers)
        torch.testing.assert_close(n1, n2)
        torch.testing.assert_close(o1["values"], o2["values"])
