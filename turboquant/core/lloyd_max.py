"""Lloyd-Max optimal scalar quantizer.

In TurboQuant, after the Hadamard rotation, each coordinate of the unit
direction vector follows a distribution dependent only on dimensionality.
This means a single fixed codebook works for ALL vectors -- no per-block
scale or zero-point metadata is needed.

The Lloyd-Max algorithm finds the codebook that minimizes mean squared error
for a given distribution. We pre-compute the optimal codebook for the
post-rotation Gaussian-like distribution (std ~ 1/sqrt(dim)).
"""

from __future__ import annotations

import torch
import math
from typing import Optional


class LloydMaxQuantizer:
    """Lloyd-Max optimal non-uniform scalar quantizer.

    For a given number of bits, computes the optimal reconstruction levels
    and decision boundaries that minimize MSE for the target distribution.
    """

    def __init__(
        self,
        bits: int = 4,
        dim: int = 128,
        max_iterations: int = 100,
        tolerance: float = 1e-7,
        device: str = "cpu",
    ):
        self.bits = bits
        self.num_levels = 1 << bits
        self.dim = dim
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.device = device

        # Pre-compute codebook for the post-rotation distribution
        self.codebook = self._compute_codebook()
        self.boundaries = self._compute_boundaries()

    def _compute_codebook(self) -> torch.Tensor:
        """Compute optimal Lloyd-Max codebook for post-Hadamard distribution.

        After Hadamard rotation, coordinates of a unit vector in R^d are
        approximately Gaussian with mean 0 and std ~ 1/sqrt(d).
        """
        std = 1.0 / math.sqrt(self.dim)
        n_levels = self.num_levels

        # Initialize with uniform quantiles of the distribution
        # Sample from the target distribution
        gen = torch.Generator(device="cpu").manual_seed(0)
        samples = torch.randn(100000, generator=gen) * std

        # Initialize codebook with evenly-spaced quantiles
        quantiles = torch.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
        codebook = torch.quantile(samples, quantiles)

        # Lloyd-Max iteration
        for _ in range(self.max_iterations):
            # Compute decision boundaries (midpoints between levels)
            boundaries = (codebook[:-1] + codebook[1:]) / 2

            # Assign samples to nearest codebook entry
            extended_boundaries = torch.cat([
                torch.tensor([float("-inf")]),
                boundaries,
                torch.tensor([float("inf")]),
            ])

            new_codebook = torch.zeros_like(codebook)
            for i in range(n_levels):
                mask = (samples >= extended_boundaries[i]) & (samples < extended_boundaries[i + 1])
                if mask.sum() > 0:
                    new_codebook[i] = samples[mask].mean()
                else:
                    new_codebook[i] = codebook[i]

            # Check convergence
            if torch.max(torch.abs(new_codebook - codebook)) < self.tolerance:
                codebook = new_codebook
                break
            codebook = new_codebook

        return codebook.to(self.device)

    def _compute_boundaries(self) -> torch.Tensor:
        """Compute decision boundaries from codebook (midpoints)."""
        return ((self.codebook[:-1] + self.codebook[1:]) / 2).to(self.device)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize values to codebook indices.

        Args:
            x: Float tensor of any shape

        Returns:
            Integer tensor of codebook indices (same shape as x)
        """
        # Compute distances to all boundaries and find bin
        x_flat = x.reshape(-1).unsqueeze(1)  # (N, 1)
        boundaries = self.boundaries.unsqueeze(0).to(x.device)  # (1, num_levels-1)

        # Each value falls into the bin where it exceeds the boundary
        indices = (x_flat >= boundaries).sum(dim=1).to(torch.int16)
        return indices.reshape(x.shape)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct values from codebook indices.

        Args:
            indices: Integer tensor of codebook indices

        Returns:
            Float tensor of reconstructed values
        """
        codebook = self.codebook.to(indices.device)
        return codebook[indices.long()]

    def quantize_and_pack(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and bit-pack for memory efficiency.

        Packs multiple sub-8-bit indices into uint8 tensors.

        Returns:
            (packed_data, shape_info) for later unpacking
        """
        indices = self.quantize(x)
        shape_info = torch.tensor(list(x.shape), dtype=torch.int64)

        if self.bits <= 4:
            # Pack two 4-bit values per byte
            flat = indices.reshape(-1)
            if flat.shape[0] % 2 != 0:
                flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])
            packed = (flat[0::2].to(torch.uint8) << 4) | flat[1::2].to(torch.uint8)
            return packed, shape_info
        elif self.bits <= 8:
            return indices.to(torch.uint8), shape_info
        else:
            return indices, shape_info

    def unpack_and_dequantize(
        self, packed: torch.Tensor, shape_info: torch.Tensor
    ) -> torch.Tensor:
        """Unpack and dequantize bit-packed tensor."""
        original_shape = tuple(shape_info.tolist())
        numel = 1
        for s in original_shape:
            numel *= s

        if self.bits <= 4:
            high = (packed >> 4).to(torch.int16)
            low = (packed & 0x0F).to(torch.int16)
            indices = torch.stack([high, low], dim=-1).reshape(-1)[:numel]
        elif self.bits <= 8:
            indices = packed.to(torch.int16).reshape(-1)[:numel]
        else:
            indices = packed.reshape(-1)[:numel]

        return self.dequantize(indices).reshape(original_shape)

    def compute_mse(self, x: torch.Tensor) -> float:
        """Compute mean squared error of quantization."""
        indices = self.quantize(x)
        reconstructed = self.dequantize(indices)
        return ((x - reconstructed) ** 2).mean().item()

    def to(self, device: str) -> LloydMaxQuantizer:
        self.device = device
        self.codebook = self.codebook.to(device)
        self.boundaries = self.boundaries.to(device)
        return self
