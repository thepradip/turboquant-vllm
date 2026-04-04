"""Quality metrics for quantization evaluation.

Comprehensive metrics suite for evaluating quantization quality at
multiple levels:
1. Element-level: MSE, SNR, relative error
2. Vector-level: cosine similarity, magnitude error
3. Attention-level: attention output error, softmax distribution divergence
4. Task-level: perplexity, accuracy on downstream tasks
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class MetricsReport:
    """Complete quality report for a quantization configuration."""
    # Element-level
    mse: float = 0.0
    snr_db: float = 0.0
    relative_error: float = 0.0

    # Vector-level
    cosine_similarity_mean: float = 0.0
    cosine_similarity_min: float = 0.0
    cosine_similarity_std: float = 0.0
    magnitude_error_mean: float = 0.0

    # Attention-level
    attention_mse: float = 0.0
    attention_cosine_similarity: float = 0.0
    kl_divergence: float = 0.0

    # Compression
    compression_ratio: float = 0.0
    effective_bits: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "snr_db": self.snr_db,
            "relative_error": self.relative_error,
            "cosine_similarity_mean": self.cosine_similarity_mean,
            "cosine_similarity_min": self.cosine_similarity_min,
            "cosine_similarity_std": self.cosine_similarity_std,
            "magnitude_error_mean": self.magnitude_error_mean,
            "attention_mse": self.attention_mse,
            "attention_cosine_similarity": self.attention_cosine_similarity,
            "kl_divergence": self.kl_divergence,
            "compression_ratio": self.compression_ratio,
            "effective_bits": self.effective_bits,
        }

    def passes_quality_threshold(
        self,
        min_cosine_sim: float = 0.95,
        max_relative_error: float = 0.1,
        min_snr_db: float = 10.0,
    ) -> bool:
        """Check if quantization meets minimum quality requirements."""
        return (
            self.cosine_similarity_mean >= min_cosine_sim
            and self.relative_error <= max_relative_error
            and self.snr_db >= min_snr_db
        )


class QualityMetrics:
    """Compute comprehensive quantization quality metrics."""

    @staticmethod
    def element_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
        """Element-level metrics: MSE, SNR, relative error."""
        diff = original - reconstructed
        mse = (diff ** 2).mean().item()
        signal_power = (original ** 2).mean().item()
        snr = 10 * math.log10(signal_power / max(mse, 1e-10))
        rel_err = (torch.norm(diff) / torch.norm(original).clamp(min=1e-8)).item()

        return {"mse": mse, "snr_db": snr, "relative_error": rel_err}

    @staticmethod
    def vector_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
        """Vector-level metrics: cosine similarity, magnitude error."""
        orig_flat = original.reshape(-1, original.shape[-1])
        recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1])

        cos_sim = F.cosine_similarity(orig_flat, recon_flat, dim=-1)
        mag_orig = torch.norm(orig_flat, dim=-1)
        mag_recon = torch.norm(recon_flat, dim=-1)
        mag_error = ((mag_orig - mag_recon).abs() / mag_orig.clamp(min=1e-8))

        return {
            "cosine_similarity_mean": cos_sim.mean().item(),
            "cosine_similarity_min": cos_sim.min().item(),
            "cosine_similarity_std": cos_sim.std().item(),
            "magnitude_error_mean": mag_error.mean().item(),
        }

    @staticmethod
    def attention_metrics(
        query: torch.Tensor,
        original_keys: torch.Tensor,
        original_values: torch.Tensor,
        quantized_keys: torch.Tensor,
        quantized_values: torch.Tensor,
        head_dim: int = 128,
    ) -> dict[str, float]:
        """Attention-level metrics: output error, softmax KL divergence."""
        scale = 1.0 / math.sqrt(head_dim)

        # Compute attention scores
        q = query.transpose(1, 2)
        ok = original_keys.transpose(1, 2)
        qk = quantized_keys.transpose(1, 2)

        orig_scores = torch.matmul(q, ok.transpose(-2, -1)) * scale
        quant_scores = torch.matmul(q, qk.transpose(-2, -1)) * scale

        orig_probs = F.softmax(orig_scores, dim=-1)
        quant_probs = F.softmax(quant_scores, dim=-1)

        # KL divergence between attention distributions
        kl_div = F.kl_div(
            quant_probs.clamp(min=1e-10).log(),
            orig_probs,
            reduction="batchmean",
        ).item()

        # Attention output error
        ov = original_values.transpose(1, 2)
        qv = quantized_values.transpose(1, 2)

        orig_output = torch.matmul(orig_probs, ov)
        quant_output = torch.matmul(quant_probs, qv)

        attn_mse = ((orig_output - quant_output) ** 2).mean().item()
        attn_cos = F.cosine_similarity(
            orig_output.reshape(-1, head_dim),
            quant_output.reshape(-1, head_dim),
            dim=-1,
        ).mean().item()

        return {
            "attention_mse": attn_mse,
            "attention_cosine_similarity": attn_cos,
            "kl_divergence": kl_div,
        }

    @classmethod
    def full_report(
        cls,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        effective_bits: float = 4.0,
    ) -> MetricsReport:
        """Generate a complete MetricsReport."""
        elem = cls.element_metrics(original, reconstructed)
        vec = cls.vector_metrics(original, reconstructed)

        return MetricsReport(
            mse=elem["mse"],
            snr_db=elem["snr_db"],
            relative_error=elem["relative_error"],
            cosine_similarity_mean=vec["cosine_similarity_mean"],
            cosine_similarity_min=vec["cosine_similarity_min"],
            cosine_similarity_std=vec["cosine_similarity_std"],
            magnitude_error_mean=vec["magnitude_error_mean"],
            compression_ratio=16.0 / effective_bits,
            effective_bits=effective_bits,
        )
