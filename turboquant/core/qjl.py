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

    def encode(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and sign-quantize residual vectors.

        Args:
            residual: Tensor of shape (..., input_dim) -- the error vectors

        Returns:
            (signs, residual_norms):
                signs: Sign-quantized projections (..., projection_dim) as {-1, +1}
                residual_norms: L2 norms of residual vectors (...,) as FP16
        """
        proj_matrix = self.projection_matrix.to(residual.device)

        # Store residual norms for proper scaling during decode
        residual_norms = torch.norm(residual, dim=-1).to(torch.float16)

        projected = residual @ proj_matrix  # (..., projection_dim)
        # Sign quantization: +1 or -1
        signs = torch.sign(projected)
        signs[signs == 0] = 1  # Map zeros to +1
        return signs, residual_norms

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

    def decode_correction(
        self, signs: torch.Tensor, residual_norms: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute approximate residual correction from sign bits.

        The correction is: R^T @ signs * scale_factor * residual_norm
        The residual norm is essential for proper magnitude reconstruction.
        Without it, the correction direction is right but magnitude is wrong.

        Args:
            signs: Sign-quantized projections of shape (..., projection_dim)
            residual_norms: L2 norms of original residuals (...,). If None,
                falls back to the old heuristic (backward compatible with 4-bit
                path that doesn't use QJL).

        Returns:
            Approximate residual correction of shape (..., input_dim)
        """
        proj_matrix = self.projection_matrix.to(signs.device)
        # Transpose projection to map back to input space
        correction = signs @ proj_matrix.T  # (..., input_dim)

        # Normalize the correction direction, then scale by residual norm
        # This is the unbiased estimator from the QJL paper:
        #   correction_hat = ||residual|| * sqrt(2/pi) * (R^T @ signs) / ||R^T @ signs||
        # sqrt(2/pi) accounts for sign quantization of Gaussian projections
        correction_norm = torch.norm(correction, dim=-1, keepdim=True).clamp(min=1e-8)
        correction = correction / correction_norm  # unit direction

        if residual_norms is not None:
            # Proper scaling: use actual residual magnitude
            scale = residual_norms.float().unsqueeze(-1) * math.sqrt(2.0 / math.pi)
        else:
            # Fallback heuristic for backward compatibility
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
