"""Tests for quality metrics suite."""

import pytest
import torch
from turboquant.utils.metrics import QualityMetrics, MetricsReport


class TestMetricsReport:
    def test_to_dict(self):
        report = MetricsReport(
            mse=0.001, snr_db=25.0, relative_error=0.03,
            cosine_similarity_mean=0.997, cosine_similarity_min=0.99,
            cosine_similarity_std=0.002, magnitude_error_mean=0.01,
            compression_ratio=4.0, effective_bits=4.0,
        )
        d = report.to_dict()
        assert d["mse"] == 0.001
        assert d["snr_db"] == 25.0
        assert d["cosine_similarity_mean"] == 0.997
        assert d["compression_ratio"] == 4.0
        assert len(d) == 12

    def test_passes_quality_threshold_pass(self):
        report = MetricsReport(
            cosine_similarity_mean=0.98,
            relative_error=0.05,
            snr_db=20.0,
        )
        assert report.passes_quality_threshold() is True

    def test_passes_quality_threshold_fail_cosine(self):
        report = MetricsReport(
            cosine_similarity_mean=0.90,  # Below 0.95 threshold
            relative_error=0.05,
            snr_db=20.0,
        )
        assert report.passes_quality_threshold() is False

    def test_passes_quality_threshold_fail_error(self):
        report = MetricsReport(
            cosine_similarity_mean=0.98,
            relative_error=0.15,  # Above 0.1 threshold
            snr_db=20.0,
        )
        assert report.passes_quality_threshold() is False

    def test_passes_quality_threshold_fail_snr(self):
        report = MetricsReport(
            cosine_similarity_mean=0.98,
            relative_error=0.05,
            snr_db=5.0,  # Below 10 dB threshold
        )
        assert report.passes_quality_threshold() is False

    def test_passes_quality_threshold_custom(self):
        report = MetricsReport(
            cosine_similarity_mean=0.85,
            relative_error=0.2,
            snr_db=5.0,
        )
        # Relaxed thresholds
        assert report.passes_quality_threshold(
            min_cosine_sim=0.80,
            max_relative_error=0.25,
            min_snr_db=3.0,
        ) is True


class TestQualityMetrics:
    def test_element_metrics(self):
        torch.manual_seed(42)
        original = torch.randn(10, 128)
        noisy = original + torch.randn_like(original) * 0.01
        m = QualityMetrics.element_metrics(original, noisy)
        assert m["mse"] > 0
        assert m["snr_db"] > 30  # Low noise = high SNR
        assert m["relative_error"] < 0.05

    def test_vector_metrics(self):
        torch.manual_seed(42)
        original = torch.randn(10, 128)
        noisy = original + torch.randn_like(original) * 0.01
        m = QualityMetrics.vector_metrics(original, noisy)
        assert m["cosine_similarity_mean"] > 0.99
        assert m["cosine_similarity_min"] > 0.95
        assert m["magnitude_error_mean"] < 0.05

    def test_attention_metrics(self):
        """Test the attention-level quality metrics (KL div, output error)."""
        torch.manual_seed(42)
        query = torch.randn(1, 4, 8, 64)
        orig_k = torch.randn(1, 16, 8, 64)
        orig_v = torch.randn(1, 16, 8, 64)
        # Small quantization noise
        quant_k = orig_k + torch.randn_like(orig_k) * 0.01
        quant_v = orig_v + torch.randn_like(orig_v) * 0.01

        m = QualityMetrics.attention_metrics(
            query, orig_k, orig_v, quant_k, quant_v, head_dim=64
        )
        assert "attention_mse" in m
        assert "attention_cosine_similarity" in m
        assert "kl_divergence" in m
        assert m["attention_cosine_similarity"] > 0.99
        assert m["kl_divergence"] >= 0

    def test_attention_metrics_large_noise(self):
        """Large quantization noise should degrade attention quality."""
        torch.manual_seed(42)
        query = torch.randn(1, 4, 8, 64)
        orig_k = torch.randn(1, 16, 8, 64)
        orig_v = torch.randn(1, 16, 8, 64)
        quant_k = orig_k + torch.randn_like(orig_k) * 1.0  # Large noise
        quant_v = orig_v + torch.randn_like(orig_v) * 1.0

        m = QualityMetrics.attention_metrics(
            query, orig_k, orig_v, quant_k, quant_v, head_dim=64
        )
        assert m["attention_cosine_similarity"] < 0.99
        assert m["kl_divergence"] > 0.01

    def test_full_report(self):
        torch.manual_seed(42)
        original = torch.randn(5, 32, 128)
        noisy = original + torch.randn_like(original) * 0.05
        report = QualityMetrics.full_report(original, noisy, effective_bits=4.0)

        assert isinstance(report, MetricsReport)
        assert report.effective_bits == 4.0
        assert report.compression_ratio == 4.0  # 16/4
        assert report.cosine_similarity_mean > 0.95
        assert report.snr_db > 10

    def test_full_report_passes_threshold(self):
        torch.manual_seed(42)
        original = torch.randn(5, 32, 128)
        noisy = original + torch.randn_like(original) * 0.01
        report = QualityMetrics.full_report(original, noisy, effective_bits=4.0)
        assert report.passes_quality_threshold() is True
