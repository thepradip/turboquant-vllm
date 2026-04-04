"""Quantized Johnson-Lindenstrauss (QJL) residual correction.

QJL is the second stage of TurboQuant, used to correct the residual error
left by PolarQuant. It works by:
1. Computing the residual: original - PolarQuant_reconstruction
2. Projecting the residual with a random JL matrix
3. Quantizing each projected coordinate to a single sign bit

The sign of each projected coordinate is an unbiased estimator of the
inner product direction. By the law of large numbers, with enough projection
dimensions, the estimate converges to the true dot product.

NOTE: Community implementations often disable QJL because it can degrade
quality in practice. It's included here for completeness and research.
"""

from __future__ import annotations

import torch
import math


class QJLProjection:
    """Quantized Johnson-Lindenstrauss projection for residual correction.

    Projects residual vectors into a lower-dimensional space and quantizes
    to sign bits, providing an unbiased estimator of inner products.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        device: str = "cpu",
        seed: int = 12345,
    ):
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.device = device

        # Generate random projection matrix (Rademacher +-1/sqrt(m))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.projection_matrix = (
            (torch.randint(0, 2, (input_dim, projection_dim), generator=gen) * 2 - 1).float()
            / math.sqrt(projection_dim)
        ).to(device)

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Project and sign-quantize residual vectors.

        Args:
            residual: Tensor of shape (..., input_dim) -- the error vectors

        Returns:
            Sign-quantized projections of shape (..., projection_dim) as {-1, +1}
        """
        proj_matrix = self.projection_matrix.to(residual.device)
        projected = residual @ proj_matrix  # (..., projection_dim)
        # Sign quantization: +1 or -1
        signs = torch.sign(projected)
        signs[signs == 0] = 1  # Map zeros to +1
        return signs

    def pack_signs(self, signs: torch.Tensor) -> torch.Tensor:
        """Pack sign bits into uint8 for storage efficiency.

        8 sign bits per byte.
        """
        flat = ((signs.reshape(-1) + 1) / 2).to(torch.uint8)  # {0, 1}
        # Pad to multiple of 8
        pad_len = (8 - flat.shape[0] % 8) % 8
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])
        # Pack 8 bits per byte
        packed = torch.zeros(flat.shape[0] // 8, dtype=torch.uint8, device=flat.device)
        for i in range(8):
            packed |= flat[i::8] << (7 - i)
        return packed

    def unpack_signs(self, packed: torch.Tensor, num_elements: int) -> torch.Tensor:
        """Unpack sign bits from uint8 storage."""
        bits = []
        for i in range(8):
            bits.append((packed >> (7 - i)) & 1)
        unpacked = torch.stack(bits, dim=-1).reshape(-1)[:num_elements]
        return unpacked.float() * 2 - 1  # {0,1} -> {-1,+1}

    def decode_correction(self, signs: torch.Tensor) -> torch.Tensor:
        """Compute approximate residual correction from sign bits.

        The correction is: R^T @ signs * scale_factor
        This provides an unbiased estimate of the original residual's
        contribution to dot products.

        Args:
            signs: Sign-quantized projections of shape (..., projection_dim)

        Returns:
            Approximate residual correction of shape (..., input_dim)
        """
        proj_matrix = self.projection_matrix.to(signs.device)
        # Transpose projection to map back to input space
        correction = signs @ proj_matrix.T  # (..., input_dim)
        # Scale factor: sqrt(projection_dim) to account for sign quantization variance
        scale = math.sqrt(self.projection_dim) / self.projection_dim
        return correction * scale

    def compute_inner_product_estimate(
        self,
        signs_a: torch.Tensor,
        signs_b: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate inner product between two vectors from their sign bits.

        By the JL lemma, <a, b> ~ (1/m) * <sign(Ra), sign(Rb)> * ||a|| * ||b||
        For sign-quantized projections, the estimator is proportional to the
        Hamming agreement between the sign vectors.
        """
        agreement = (signs_a * signs_b).sum(dim=-1)
        return agreement / self.projection_dim

    def to(self, device: str) -> QJLProjection:
        self.device = device
        self.projection_matrix = self.projection_matrix.to(device)
        return self
