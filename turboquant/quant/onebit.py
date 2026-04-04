"""1-bit quantization inspired by PrismML's Bonsai Q1_0_g128 format.

The Q1_0_g128 format represents each weight as a single sign bit:
- 0 maps to -scale
- 1 maps to +scale

Every 128 consecutive weights share one FP16 scale factor, computed as
the mean absolute value of the group. This gives an effective bit-width
of 1.125 bits per weight (1 sign bit + 16-bit scale amortized over 128).

This achieves 93% reduction in parameter memory compared to FP16 and
can be applied to KV cache entries for extreme compression scenarios.
"""

from __future__ import annotations

import torch
import math


class OneBitQuantizer:
    """Bonsai-inspired 1-bit quantization with grouped scales.

    Format: Q1_0_g{group_size}
    - Each weight stored as 1 sign bit
    - Groups of `group_size` weights share one FP16 scale
    - Effective bits per weight: 1 + 16/group_size
    """

    def __init__(self, group_size: int = 128, device: str = "cpu"):
        if group_size < 1:
            raise ValueError(f"Group size must be positive, got {group_size}")
        self.group_size = group_size
        self.device = device
        self.effective_bits = 1.0 + 16.0 / group_size

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to 1-bit with grouped scales.

        Args:
            x: Float tensor of shape (..., dim) where dim should be
               divisible by group_size for optimal results

        Returns:
            (packed_signs, scales):
                packed_signs: uint8 tensor with 8 sign bits per byte
                scales: FP16 tensor with one scale per group
        """
        orig_shape = x.shape
        dim = x.shape[-1]

        # Reshape into groups
        x_flat = x.reshape(-1, dim)
        batch = x_flat.shape[0]

        # Pad dim to multiple of group_size
        pad_size = (self.group_size - dim % self.group_size) % self.group_size
        if pad_size > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_size))

        padded_dim = x_flat.shape[-1]
        num_groups = padded_dim // self.group_size

        # Reshape to (batch, num_groups, group_size)
        grouped = x_flat.reshape(batch, num_groups, self.group_size)

        # Compute per-group scales (mean absolute value)
        scales = grouped.abs().mean(dim=-1).to(torch.float16)  # (batch, num_groups)

        # Extract sign bits: 1 for positive/zero, 0 for negative
        signs = (grouped >= 0).to(torch.uint8)  # (batch, num_groups, group_size)

        # Pack 8 sign bits per byte
        signs_flat = signs.reshape(batch, -1)  # (batch, padded_dim)
        packed = self._pack_bits(signs_flat)

        return packed, scales

    def dequantize(
        self, packed: torch.Tensor, scales: torch.Tensor, original_dim: int
    ) -> torch.Tensor:
        """Dequantize from 1-bit packed format.

        Args:
            packed: uint8 tensor of packed sign bits
            scales: FP16 tensor of per-group scales
            original_dim: original dimension before padding

        Returns:
            Reconstructed float tensor
        """
        batch = scales.shape[0]
        num_groups = scales.shape[1]
        padded_dim = num_groups * self.group_size

        # Unpack sign bits
        signs = self._unpack_bits(packed, padded_dim)  # (batch, padded_dim)
        signs = signs.reshape(batch, num_groups, self.group_size)

        # Convert signs: 0 -> -1, 1 -> +1
        sign_values = signs.float() * 2 - 1  # {-1, +1}

        # Scale by group scale: x_reconstructed = sign * scale
        scales_expanded = scales.float().unsqueeze(-1)  # (batch, num_groups, 1)
        reconstructed = sign_values * scales_expanded

        # Reshape and remove padding
        reconstructed = reconstructed.reshape(batch, padded_dim)
        if padded_dim != original_dim:
            reconstructed = reconstructed[:, :original_dim]

        return reconstructed

    def _pack_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Pack boolean/uint8 bits into uint8 bytes (8 bits per byte).

        Args:
            bits: (batch, dim) tensor of 0/1 values

        Returns:
            (batch, ceil(dim/8)) tensor of packed uint8
        """
        batch, dim = bits.shape
        # Pad to multiple of 8
        pad = (8 - dim % 8) % 8
        if pad > 0:
            bits = torch.nn.functional.pad(bits, (0, pad))
        packed_dim = bits.shape[1] // 8

        bits = bits.reshape(batch, packed_dim, 8)
        packed = torch.zeros(batch, packed_dim, dtype=torch.uint8, device=bits.device)
        for i in range(8):
            packed |= bits[:, :, i].to(torch.uint8) << (7 - i)
        return packed

    def _unpack_bits(self, packed: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Unpack uint8 bytes into individual bits.

        Args:
            packed: (batch, packed_dim) uint8 tensor
            num_bits: total number of bits to unpack

        Returns:
            (batch, num_bits) tensor of 0/1 values
        """
        batch = packed.shape[0]
        bits = []
        for i in range(8):
            bits.append((packed >> (7 - i)) & 1)
        unpacked = torch.stack(bits, dim=-1).reshape(batch, -1)
        return unpacked[:, :num_bits]

    def compute_quality(self, original: torch.Tensor) -> dict[str, float]:
        """Compute quantization quality metrics."""
        packed, scales = self.quantize(original)
        dim = original.shape[-1]
        x_flat = original.reshape(-1, dim)
        reconstructed = self.dequantize(packed, scales, dim)

        mse = ((x_flat - reconstructed) ** 2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(x_flat, reconstructed, dim=-1)
        snr = 10 * math.log10(
            (x_flat ** 2).mean().item() / max(mse, 1e-10)
        )

        return {
            "mse": mse,
            "cosine_similarity_mean": cos_sim.mean().item(),
            "cosine_similarity_min": cos_sim.min().item(),
            "snr_db": snr,
            "effective_bits": self.effective_bits,
            "compression_ratio": 16.0 / self.effective_bits,
            "memory_reduction_pct": (1 - self.effective_bits / 16) * 100,
        }
