"""Fast quantizer optimized for Metal (MPS) and CUDA (T4+).

Drop-in replacement for TurboQuantizer with:
- torch.bucketize instead of loop-based codebook lookup
- Fused encode/decode (fewer kernel launches)
- torch.compile support for automatic kernel fusion
- Device-aware: auto-selects MPS/CUDA/CPU

Usage:
    from turboquant.core.fast_quantizer import FastTurboQuantizer
    q = FastTurboQuantizer(config)  # same API as TurboQuantizer
"""

from __future__ import annotations

import torch
import math
from typing import Optional

from turboquant.config import TurboQuantConfig
from turboquant.core.lloyd_max import LloydMaxQuantizer
from turboquant.core import fast_ops


class FastTurboQuantizer:
    """High-performance quantizer for Metal/CUDA.

    Same API as TurboQuantizer. Key speedups:
    1. torch.bucketize for codebook quantization (~10x vs loop)
    2. Fused encode: magnitude + hadamard + quantize in one pass
    3. Pre-allocated codebook/signs on device (no transfers)
    4. torch.compile-compatible (no Python control flow in hot path)
    """

    def __init__(self, config: TurboQuantConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or self._auto_device()

        # Pre-compute and move to device
        self._init_codebook()
        self._init_signs()

    def _auto_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_codebook(self) -> None:
        """Pre-compute Lloyd-Max codebook and move to device."""
        dim = self.config.head_dim
        self._padded_dim = 1 << (dim - 1).bit_length() if dim & (dim - 1) != 0 else dim

        # Build codebook for keys
        lm_k = LloydMaxQuantizer(
            bits=self.config.effective_key_bits, dim=self._padded_dim
        )
        self._codebook_k = lm_k.codebook.to(self.device)
        self._boundaries_k = lm_k.boundaries.to(self.device)

        # Build codebook for values
        lm_v = LloydMaxQuantizer(
            bits=self.config.effective_value_bits, dim=self._padded_dim
        )
        self._codebook_v = lm_v.codebook.to(self.device)
        self._boundaries_v = lm_v.boundaries.to(self.device)

    def _init_signs(self) -> None:
        """Pre-compute Hadamard random signs on device."""
        gen = torch.Generator(device="cpu").manual_seed(42)
        self._signs_k = (
            (torch.randint(0, 2, (self._padded_dim,), generator=gen) * 2 - 1)
            .float().to(self.device)
        )
        gen2 = torch.Generator(device="cpu").manual_seed(43)
        self._signs_v = (
            (torch.randint(0, 2, (self._padded_dim,), generator=gen2) * 2 - 1)
            .float().to(self.device)
        )

    def encode_keys(
        self, keys: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Fast encode keys."""
        if self.config.enable_polar_quant:
            return self._fast_polar_encode(keys, self._signs_k, self._boundaries_k, is_key=True)
        if self.config.enable_onebit:
            return self._fast_onebit_encode(keys)
        return self._fast_asymmetric_encode(keys, is_key=True)

    def encode_values(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Fast encode values."""
        if self.config.enable_polar_quant:
            return self._fast_polar_encode(values, self._signs_v, self._boundaries_v, is_key=False)
        if self.config.enable_onebit:
            return self._fast_onebit_encode(values)
        return self._fast_asymmetric_encode(values, is_key=False)

    def decode_keys(
        self, quantized: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        """Fast decode keys."""
        if self.config.enable_polar_quant:
            return self._fast_polar_decode(quantized, meta, self._codebook_k, self._signs_k)
        if self.config.enable_onebit:
            return self._fast_onebit_decode(quantized, meta)
        return self._fast_asymmetric_decode(quantized, meta)

    def decode_values(
        self, quantized: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        """Fast decode values."""
        if self.config.enable_polar_quant:
            return self._fast_polar_decode(quantized, meta, self._codebook_v, self._signs_v)
        if self.config.enable_onebit:
            return self._fast_onebit_decode(quantized, meta)
        return self._fast_asymmetric_decode(quantized, meta)

    def _fast_polar_encode(
        self, x: torch.Tensor, signs: torch.Tensor, boundaries: torch.Tensor, is_key: bool
    ) -> tuple[torch.Tensor, dict]:
        batch, seq, heads, dim = x.shape
        x_flat = x.reshape(-1, dim)

        magnitudes, indices, _ = fast_ops.fast_polar_encode(
            x_flat, signs, boundaries, self._padded_dim
        )

        meta = {"magnitudes": magnitudes.reshape(batch, seq, heads)}
        return indices.reshape(batch, seq, heads, -1), meta

    def _fast_polar_decode(
        self, quantized: torch.Tensor, meta: dict,
        codebook: torch.Tensor, signs: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, heads = quantized.shape[:3]
        indices_flat = quantized.reshape(-1, quantized.shape[-1])
        magnitudes = meta["magnitudes"].reshape(-1)

        reconstructed = fast_ops.fast_polar_decode(
            magnitudes, indices_flat, codebook, signs,
            self.config.head_dim, self._padded_dim
        )
        return reconstructed.reshape(batch, seq, heads, -1)

    def _fast_asymmetric_encode(
        self, x: torch.Tensor, is_key: bool
    ) -> tuple[torch.Tensor, dict]:
        bits = self.config.effective_key_bits if is_key else self.config.effective_value_bits
        if is_key and self.config.key_quant_mode == "per_channel":
            q, scale, zero = fast_ops.fast_per_channel_quantize(x, bits)
        else:
            q, scale, zero = fast_ops.fast_per_token_quantize(x, bits)
        return q, {"scales": scale, "zeros": zero}

    def _fast_asymmetric_decode(
        self, quantized: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        return fast_ops.fast_dequantize(quantized, meta["scales"], meta["zeros"])

    def _fast_onebit_encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        batch, seq, heads, dim = x.shape
        x_flat = x.reshape(-1, dim)

        # Apply Hadamard if enabled
        if self.config.enable_hadamard:
            if dim != self._padded_dim:
                x_flat = torch.nn.functional.pad(x_flat, (0, self._padded_dim - dim))
            x_flat = fast_ops.fast_hadamard_inplace(x_flat.clone(), self._signs_k)

        packed, scales = fast_ops.fast_onebit_encode(x_flat, self.config.onebit_group_size)
        meta = {
            "scales": scales.reshape(batch, seq, heads, -1),
            "original_shape": torch.tensor([batch, seq, heads, dim]),
            "hadamard_applied": self.config.enable_hadamard,
        }
        return packed.reshape(batch, seq, heads, -1), meta

    def _fast_onebit_decode(
        self, quantized: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        if "original_shape" in meta:
            batch, seq, heads, dim = tuple(meta["original_shape"].tolist())
        else:
            batch, seq, heads = quantized.shape[:3]
            dim = self.config.head_dim

        packed_flat = quantized.reshape(-1, quantized.shape[-1])
        scales = meta["scales"].reshape(-1, meta["scales"].shape[-1])

        hadamard_applied = meta.get("hadamard_applied", False)
        dequant_dim = self._padded_dim if hadamard_applied else dim

        reconstructed = fast_ops.fast_onebit_decode(
            packed_flat, scales, dequant_dim, self.config.onebit_group_size
        )

        if hadamard_applied:
            reconstructed = fast_ops.fast_hadamard_inplace(reconstructed.clone(), self._signs_k)
            if dequant_dim != dim:
                reconstructed = reconstructed[:, :dim]

        return reconstructed.reshape(batch, seq, heads, dim)

    def compute_compression_ratio(self, original_bytes: int) -> dict[str, float]:
        bits = self.config.effective_key_bits
        if self.config.enable_onebit:
            effective_bits = 1.0 + 16.0 / self.config.onebit_group_size
        elif self.config.enable_polar_quant:
            effective_bits = bits + 16.0 / self.config.head_dim
        else:
            effective_bits = bits
        ratio = 16.0 / effective_bits
        return {
            "effective_bits_per_element": effective_bits,
            "compression_ratio": ratio,
            "original_bytes": original_bytes,
            "compressed_bytes": original_bytes / ratio,
            "memory_savings_pct": (1 - 1 / ratio) * 100,
        }

    def to(self, device: str) -> FastTurboQuantizer:
        self.device = device
        self._codebook_k = self._codebook_k.to(device)
        self._codebook_v = self._codebook_v.to(device)
        self._boundaries_k = self._boundaries_k.to(device)
        self._boundaries_v = self._boundaries_v.to(device)
        self._signs_k = self._signs_k.to(device)
        self._signs_v = self._signs_v.to(device)
        return self
