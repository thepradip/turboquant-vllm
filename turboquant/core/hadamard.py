"""Hadamard rotation transforms for outlier elimination.

Random Hadamard rotations redistribute outlier magnitudes uniformly across
all channels, making the distribution more amenable to quantization. This
is a key insight from QuaRot and TurboQuant -- the rotation is an orthogonal
transform that preserves dot products and vector norms while eliminating
the channel-wise outlier patterns that plague naive quantization.

The Fast Walsh-Hadamard Transform runs in O(d log d) vs O(d^2) for a full
matrix multiply, making it practical for online inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math


class HadamardTransform:
    """Fast Walsh-Hadamard transform with optional random sign flipping.

    The random diagonal sign matrix D converts the deterministic Hadamard
    matrix H into a randomized orthogonal transform (H @ D), which provides
    the universal distribution guarantee that TurboQuant's PolarQuant relies on:
    after rotation, each coordinate follows a distribution dependent only on
    dimensionality, not on the input vector.
    """

    def __init__(self, dim: int, device: str = "cpu", seed: int = 42):
        if dim < 1 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Dimension must be a power of 2, got {dim}")
        self.dim = dim
        self.device = device
        # Random sign vector for randomized Hadamard
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.signs = (torch.randint(0, 2, (dim,), generator=gen) * 2 - 1).float().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomized Hadamard transform: H @ diag(signs) @ x.

        Uses the Fast Walsh-Hadamard Transform (butterfly structure) for
        O(d log d) complexity.

        Args:
            x: Tensor of shape (..., dim)

        Returns:
            Transformed tensor of same shape, normalized by 1/sqrt(dim)
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        # Apply random signs
        y = x * self.signs.to(x.device)
        # Fast Walsh-Hadamard butterfly
        y = self._fwht(y)
        return y

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse randomized Hadamard: diag(signs) @ H^T @ x.

        Since H is symmetric and orthogonal (H^T = H, H @ H = d * I),
        the inverse is H @ x / d, then multiply by signs.
        But since we normalize by 1/sqrt(d), the inverse is just
        applying the same transform again and multiplying by signs.
        """
        y = self._fwht(x)
        y = y * self.signs.to(x.device)
        return y

    def _fwht(self, x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh-Hadamard Transform via butterfly operations.

        Operates in-place on the last dimension. The butterfly structure
        computes the full d-dimensional Hadamard in log2(d) stages, each
        performing d/2 additions and subtractions.
        """
        orig_shape = x.shape
        n = x.shape[-1]

        if n != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {n}")

        # Reshape for butterfly operations
        x = x.clone().reshape(-1, n)
        batch = x.shape[0]

        h = 1
        while h < n:
            # Split into pairs at stride h
            x_view = x.view(batch, n // (2 * h), 2, h)
            a = x_view[:, :, 0, :].clone()
            b = x_view[:, :, 1, :].clone()
            x_view[:, :, 0, :] = a + b
            x_view[:, :, 1, :] = a - b
            x = x_view.reshape(batch, n)
            h *= 2

        # Normalize
        x = x / math.sqrt(n)
        return x.reshape(orig_shape)

    def to(self, device: str) -> HadamardTransform:
        self.device = device
        self.signs = self.signs.to(device)
        return self


def pad_to_power_of_2(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, int]:
    """Pad tensor's specified dimension to the next power of 2."""
    size = x.shape[dim]
    if size & (size - 1) == 0:
        return x, size
    next_pow2 = 1 << (size - 1).bit_length()
    pad_size = next_pow2 - size
    pad_dims = [0] * (2 * (x.ndim - 1 - (dim % x.ndim))) + [0, pad_size]
    return F.pad(x, pad_dims), next_pow2


def unpad(x: torch.Tensor, original_size: int, dim: int = -1) -> torch.Tensor:
    """Remove padding from tensor along specified dimension."""
    return x.narrow(dim, 0, original_size)
