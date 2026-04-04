"""PolarQuant: the primary compression stage of TurboQuant.

PolarQuant decomposes each KV vector into:
1. Magnitude (scalar radius r) -- stored in FP16
2. Direction (unit vector on the hypersphere) -- quantized via Lloyd-Max

The direction vector is first rotated by a randomized Hadamard transform,
which redistributes coordinates to follow a predictable distribution
(approximately Gaussian with std ~ 1/sqrt(dim)). This universal distribution
means a single fixed Lloyd-Max codebook applies to ALL vectors.

At 4-bit, PolarQuant achieves near-lossless compression with high cosine
similarity to the original vectors.
"""

from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import Optional

from turboquant.core.hadamard import HadamardTransform, pad_to_power_of_2, unpad
from turboquant.core.lloyd_max import LloydMaxQuantizer


@dataclass
class PolarQuantOutput:
    """Output of PolarQuant encoding."""
    magnitudes: torch.Tensor       # (batch, seq, heads) -- FP16 scalars
    quantized_indices: torch.Tensor  # (batch, seq, heads, dim) -- int codebook indices
    packed_data: Optional[torch.Tensor] = None  # Bit-packed version
    shape_info: Optional[torch.Tensor] = None
    original_dim: int = 0  # Before padding


class PolarQuant:
    """PolarQuant vector quantization for KV cache compression.

    Decomposes vectors into magnitude + direction, applies Hadamard rotation
    to the direction, then quantizes with a fixed Lloyd-Max codebook.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.original_dim = dim
        # Pad to power of 2 for Hadamard
        self.padded_dim = 1 << (dim - 1).bit_length() if dim & (dim - 1) != 0 else dim
        self.bits = bits
        self.device = device

        self.hadamard = HadamardTransform(self.padded_dim, device=device, seed=seed)
        self.lloyd_max = LloydMaxQuantizer(
            bits=bits, dim=self.padded_dim, device=device
        )

    def encode(self, x: torch.Tensor, pack: bool = False) -> PolarQuantOutput:
        """Encode vectors using PolarQuant.

        Args:
            x: Tensor of shape (..., dim) -- the KV vectors to compress
            pack: Whether to bit-pack the quantized indices

        Returns:
            PolarQuantOutput with magnitudes and quantized direction indices
        """
        original_shape = x.shape
        dim = x.shape[-1]

        # Step 1: Decompose into magnitude and direction
        magnitudes = torch.norm(x, dim=-1, keepdim=True)  # (..., 1)
        # Avoid division by zero
        safe_mag = magnitudes.clamp(min=1e-8)
        directions = x / safe_mag  # (..., dim) unit vectors

        # Step 2: Pad to power of 2 if needed
        if dim != self.padded_dim:
            pad_size = self.padded_dim - dim
            directions = torch.nn.functional.pad(directions, (0, pad_size))

        # Step 3: Apply randomized Hadamard rotation
        rotated = self.hadamard.forward(directions)

        # Step 4: Quantize rotated coordinates with Lloyd-Max codebook
        indices = self.lloyd_max.quantize(rotated)

        output = PolarQuantOutput(
            magnitudes=magnitudes.squeeze(-1).to(torch.float16),
            quantized_indices=indices,
            original_dim=dim,
        )

        if pack:
            packed, shape_info = self.lloyd_max.quantize_and_pack(rotated)
            output.packed_data = packed
            output.shape_info = shape_info

        return output

    def decode(self, encoded: PolarQuantOutput) -> torch.Tensor:
        """Decode PolarQuant-compressed vectors back to full precision.

        Args:
            encoded: PolarQuantOutput from encode()

        Returns:
            Reconstructed tensor of original shape (..., dim)
        """
        # Step 1: Dequantize from codebook
        if encoded.packed_data is not None and encoded.shape_info is not None:
            rotated_reconstructed = self.lloyd_max.unpack_and_dequantize(
                encoded.packed_data, encoded.shape_info
            )
        else:
            rotated_reconstructed = self.lloyd_max.dequantize(encoded.quantized_indices)

        # Step 2: Inverse Hadamard rotation
        directions_reconstructed = self.hadamard.inverse(rotated_reconstructed)

        # Step 3: Remove padding
        if encoded.original_dim != self.padded_dim:
            directions_reconstructed = directions_reconstructed[..., :encoded.original_dim]

        # Step 4: Re-normalize to unit vector (rotation + quantization may break unit norm)
        norm = torch.norm(directions_reconstructed, dim=-1, keepdim=True).clamp(min=1e-8)
        directions_reconstructed = directions_reconstructed / norm

        # Step 5: Scale by original magnitude
        magnitudes = encoded.magnitudes.float().unsqueeze(-1)
        return directions_reconstructed * magnitudes

    def compute_quality_metrics(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> dict[str, float]:
        """Compute quality metrics between original and reconstructed vectors."""
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original.reshape(-1, original.shape[-1]),
            reconstructed.reshape(-1, reconstructed.shape[-1]),
            dim=-1,
        )

        # Relative error
        rel_error = torch.norm(original - reconstructed, dim=-1) / (
            torch.norm(original, dim=-1).clamp(min=1e-8)
        )

        # MSE
        mse = ((original - reconstructed) ** 2).mean()

        return {
            "cosine_similarity_mean": cos_sim.mean().item(),
            "cosine_similarity_min": cos_sim.min().item(),
            "cosine_similarity_std": cos_sim.std().item(),
            "relative_error_mean": rel_error.mean().item(),
            "relative_error_max": rel_error.max().item(),
            "mse": mse.item(),
        }

    def to(self, device: str) -> PolarQuant:
        self.device = device
        self.hadamard = self.hadamard.to(device)
        self.lloyd_max = self.lloyd_max.to(device)
        return self
