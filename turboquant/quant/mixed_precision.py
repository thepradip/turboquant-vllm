"""Mixed-precision quantization with outlier channel handling.

Some channels in key/value matrices exhibit persistent outliers --
magnitudes 10-100x larger than typical channels. These outlier channels
carry disproportionate information and are highly sensitive to quantization.

Strategy:
- Identify the top-k outlier channels by magnitude
- Keep outlier channels at higher precision (8-bit or FP16)
- Quantize remaining channels at the target low bit-width
- This gives a fractional average bit-width (e.g., 25% at 8-bit + 75% at 4-bit = 5-bit avg)
"""

from __future__ import annotations

import torch
from typing import Optional


class MixedPrecisionQuantizer:
    """Handles mixed-precision quantization by separating outlier channels."""

    def __init__(
        self,
        outlier_fraction: float = 0.25,
        outlier_bits: int = 8,
        normal_bits: int = 4,
        device: str = "cpu",
    ):
        self.outlier_fraction = outlier_fraction
        self.outlier_bits = outlier_bits
        self.normal_bits = normal_bits
        self.device = device
        self._outlier_indices: Optional[torch.Tensor] = None
        self._normal_indices: Optional[torch.Tensor] = None

    @property
    def effective_bits(self) -> float:
        f = self.outlier_fraction
        return f * self.outlier_bits + (1 - f) * self.normal_bits

    def identify_outlier_channels(
        self, x: torch.Tensor, calibration_tokens: int = 0
    ) -> torch.Tensor:
        """Identify outlier channels by magnitude.

        Channels are ranked by their L2 norm across all tokens. The top
        `outlier_fraction` channels are marked as outliers.

        Args:
            x: (num_vectors, dim)

        Returns:
            Tensor of outlier channel indices
        """
        dim = x.shape[-1]
        num_outliers = max(1, int(dim * self.outlier_fraction))

        # Compute per-channel magnitude (L2 norm across vectors)
        channel_norms = torch.norm(x, dim=0)  # (dim,)

        # Top-k outlier channels
        _, outlier_idx = torch.topk(channel_norms, num_outliers)
        self._outlier_indices = outlier_idx.sort().values

        # Normal channels
        all_idx = torch.arange(dim, device=x.device)
        mask = torch.ones(dim, dtype=torch.bool, device=x.device)
        mask[self._outlier_indices] = False
        self._normal_indices = all_idx[mask]

        return self._outlier_indices

    def extract_outliers(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extract outlier channels from input, return normal channels + outlier data.

        If outlier indices haven't been identified yet, identifies them now.

        Args:
            x: (num_vectors, dim)

        Returns:
            (normal_channels, {"indices": outlier_idx, "values": outlier_values})
        """
        if self._outlier_indices is None:
            self.identify_outlier_channels(x)

        outlier_idx = self._outlier_indices.to(x.device)
        normal_idx = self._normal_indices.to(x.device)

        outlier_values = x[:, outlier_idx]  # (N, num_outliers)

        # Zero out outlier channels in x (will be quantized at lower precision)
        x_normal = x.clone()
        x_normal[:, outlier_idx] = 0

        return x_normal, {
            "indices": outlier_idx,
            "values": outlier_values,
        }

    def restore_outliers(
        self,
        x_normal: torch.Tensor,
        outlier_indices: torch.Tensor,
        outlier_values: torch.Tensor,
    ) -> torch.Tensor:
        """Restore outlier channels into the dequantized tensor."""
        outlier_indices = outlier_indices.to(x_normal.device)
        outlier_values = outlier_values.to(x_normal.device)
        result = x_normal.clone()
        result[:, outlier_indices] = outlier_values
        return result

    def quantize_outliers_int8(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize outlier channels to INT8.

        Args:
            values: (N, num_outlier_channels)

        Returns:
            (quantized_int8, scales, zero_points)
        """
        vmin = values.amin(dim=0, keepdim=True)
        vmax = values.amax(dim=0, keepdim=True)
        vrange = (vmax - vmin).clamp(min=1e-8)
        scale = vrange / 255.0
        zero_point = vmin
        quantized = ((values - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)
        return quantized, scale, zero_point

    def dequantize_outliers_int8(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize INT8 outlier channels."""
        return quantized.float() * scale + zero_point
