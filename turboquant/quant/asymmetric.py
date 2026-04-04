"""KIVI-style asymmetric quantization.

Key insight from the KIVI paper (ICML 2024):
- Keys have persistent outlier channels (same channels are large across all tokens)
  -> Per-CHANNEL quantization confines error to outlier channels
- Values have no obvious outlier pattern, and attention is sparse
  -> Per-TOKEN quantization confines error to individual token representations

This asymmetric approach yields 5x lower attention score error for keys
and 15x lower error for values compared to the naive inverse strategies.

From KV-AdaQuant: keys have 10-50x higher quantization error at identical
bit-widths due to larger spectral/Frobenius norms, so we default to giving
keys more bits than values (4-bit keys + 2-bit values).
"""

from __future__ import annotations

import torch
from typing import Optional


class AsymmetricQuantizer:
    """KIVI-style asymmetric quantization with per-channel keys and per-token values."""

    def __init__(
        self,
        key_bits: int = 4,
        value_bits: int = 2,
        key_mode: str = "per_channel",
        value_mode: str = "per_token",
        group_size: int = 32,
        device: str = "cpu",
    ):
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.key_mode = key_mode
        self.value_mode = value_mode
        self.group_size = group_size
        self.device = device
        self.key_qmax = (1 << key_bits) - 1
        self.value_qmax = (1 << value_bits) - 1

    def quantize_keys(
        self, keys: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Quantize keys using per-channel strategy.

        Per-channel: quantization parameters are computed along the token
        dimension for each channel independently. This isolates outlier
        channels so their large range doesn't affect other channels.

        Args:
            keys: (batch, seq_len, num_kv_heads, head_dim)

        Returns:
            (quantized_keys, {"scales": ..., "zeros": ...})
        """
        if self.key_mode == "per_channel":
            return self._quantize_per_channel(keys, self.key_bits, self.key_qmax)
        else:
            return self._quantize_per_token(keys, self.key_bits, self.key_qmax)

    def quantize_values(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Quantize values using per-token strategy.

        Per-token: quantization parameters are computed for each token
        independently across channels. Since attention is sparse, errors
        are confined to tokens that contribute little to the output.

        Args:
            values: (batch, seq_len, num_kv_heads, head_dim)

        Returns:
            (quantized_values, {"scales": ..., "zeros": ...})
        """
        if self.value_mode == "per_token":
            return self._quantize_per_token(values, self.value_bits, self.value_qmax)
        else:
            return self._quantize_per_channel(values, self.value_bits, self.value_qmax)

    def _quantize_per_channel(
        self, x: torch.Tensor, bits: int, qmax: int
    ) -> tuple[torch.Tensor, dict]:
        """Per-channel (asymmetric min-max) quantization.

        Computes scale and zero-point for each channel across all tokens.
        Shape: x is (batch, seq, heads, dim)
        Scales/zeros have shape (batch, 1, heads, dim) -- one per channel.
        """
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)

        # Avoid zero range
        x_range = (x_max - x_min).clamp(min=1e-8)
        scale = x_range / qmax
        zero_point = x_min

        quantized = ((x - zero_point) / scale).round().clamp(0, qmax).to(torch.int16)

        return quantized, {"scales": scale, "zeros": zero_point}

    def _quantize_per_token(
        self, x: torch.Tensor, bits: int, qmax: int
    ) -> tuple[torch.Tensor, dict]:
        """Per-token (asymmetric min-max) quantization.

        Computes scale and zero-point for each token across all channels.
        Shape: x is (batch, seq, heads, dim)
        Scales/zeros have shape (batch, seq, heads, 1) -- one per token per head.
        """
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)

        x_range = (x_max - x_min).clamp(min=1e-8)
        scale = x_range / qmax
        zero_point = x_min

        quantized = ((x - zero_point) / scale).round().clamp(0, qmax).to(torch.int16)

        return quantized, {"scales": scale, "zeros": zero_point}

    def dequantize_keys(
        self, quantized: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        """Dequantize keys."""
        scales = meta["scales"]
        zeros = meta["zeros"]
        if scales is None or zeros is None:
            return quantized.float()
        return quantized.float() * scales + zeros

    def dequantize_values(
        self, quantized: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        """Dequantize values."""
        scales = meta["scales"]
        zeros = meta["zeros"]
        if scales is None or zeros is None:
            return quantized.float()
        return quantized.float() * scales + zeros

    def compute_error(
        self, original: torch.Tensor, quantized: torch.Tensor, meta: dict
    ) -> dict[str, float]:
        """Compute quantization error metrics."""
        reconstructed = quantized.float() * meta["scales"] + meta["zeros"]
        mse = ((original - reconstructed) ** 2).mean().item()
        rel_err = (
            torch.norm(original - reconstructed)
            / torch.norm(original).clamp(min=1e-8)
        ).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            original.reshape(-1, original.shape[-1]),
            reconstructed.reshape(-1, original.shape[-1]),
            dim=-1,
        ).mean().item()

        return {
            "mse": mse,
            "relative_error": rel_err,
            "cosine_similarity": cos_sim,
        }
