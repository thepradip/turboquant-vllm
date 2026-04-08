"""Main TurboQuantizer: orchestrates all quantization stages.

This is the central module that combines:
- PolarQuant (magnitude + rotated direction quantization)
- QJL residual correction (optional)
- KIVI asymmetric quantization (per-channel keys, per-token values)
- Mixed-precision outlier handling
- 1-bit Bonsai-style quantization

The quantizer provides encode/decode methods for both keys and values,
with different strategies applied based on configuration.
"""

from __future__ import annotations

import torch
from typing import Optional

from turboquant.config import TurboQuantConfig
from turboquant.core.polar_quant import PolarQuant
from turboquant.core.qjl import QJLProjection
from turboquant.core.hadamard import HadamardTransform, pad_to_power_of_2, unpad
from turboquant.quant.asymmetric import AsymmetricQuantizer
from turboquant.quant.onebit import OneBitQuantizer
from turboquant.quant.mixed_precision import MixedPrecisionQuantizer


class TurboQuantizer:
    """Orchestrates the full TurboQuant quantization pipeline.

    Encoding pipeline (for each KV vector):
    1. [Optional] Identify outlier channels -> split into outlier + normal
    2. [Optional] Apply Hadamard rotation to normal channels
    3. Quantize using configured strategy:
       - PolarQuant: magnitude/direction decomposition + Lloyd-Max
       - KIVI: per-channel (keys) / per-token (values) uniform quantization
       - 1-bit: Bonsai-style Q1_0_g128
    4. [Optional] QJL residual correction on PolarQuant output
    5. Pack quantized data for memory efficiency

    Decoding pipeline reverses the above.
    """

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.device = config.device

        # Initialize components based on config
        self._init_polar_quant()
        self._init_qjl()
        self._init_hadamard()
        self._init_asymmetric()
        self._init_onebit()
        self._init_mixed_precision()

    def _init_polar_quant(self) -> None:
        self.polar_quant_key: Optional[PolarQuant] = None
        self.polar_quant_value: Optional[PolarQuant] = None
        if self.config.enable_polar_quant:
            self.polar_quant_key = PolarQuant(
                dim=self.config.head_dim,
                bits=self.config.effective_key_bits,
                device=self.device,
                seed=42,
            )
            self.polar_quant_value = PolarQuant(
                dim=self.config.head_dim,
                bits=self.config.effective_value_bits,
                device=self.device,
                seed=43,  # Different seed for values
            )

    def _init_qjl(self) -> None:
        self.qjl: Optional[QJLProjection] = None
        if self.config.enable_qjl:
            self.qjl = QJLProjection(
                input_dim=self.config.head_dim,
                projection_dim=self.config.qjl_projection_dim,
                device=self.device,
            )

    def _init_asymmetric(self) -> None:
        self.asymmetric: Optional[AsymmetricQuantizer] = None
        if self.config.enable_asymmetric and not self.config.enable_polar_quant:
            self.asymmetric = AsymmetricQuantizer(
                key_bits=self.config.effective_key_bits,
                value_bits=self.config.effective_value_bits,
                key_mode=self.config.key_quant_mode,
                value_mode=self.config.value_quant_mode,
                group_size=self.config.kivi_group_size,
                device=self.device,
            )

    def _init_onebit(self) -> None:
        self.onebit: Optional[OneBitQuantizer] = None
        if self.config.enable_onebit:
            self.onebit = OneBitQuantizer(
                group_size=self.config.onebit_group_size,
                device=self.device,
            )

    def _init_hadamard(self) -> None:
        """Standalone Hadamard for 1-bit and other non-PolarQuant paths."""
        self.hadamard: Optional[HadamardTransform] = None
        if self.config.enable_hadamard and not self.config.enable_polar_quant:
            dim = self.config.head_dim
            padded = 1 << (dim - 1).bit_length() if dim & (dim - 1) != 0 else dim
            self.hadamard = HadamardTransform(padded, device=self.device, seed=42)
            self._hadamard_padded_dim = padded
            self._hadamard_original_dim = dim

    def _init_mixed_precision(self) -> None:
        self.mixed_precision: Optional[MixedPrecisionQuantizer] = None
        if self.config.enable_mixed_precision:
            self.mixed_precision = MixedPrecisionQuantizer(
                outlier_fraction=self.config.outlier_channel_fraction,
                outlier_bits=self.config.outlier_bits,
                normal_bits=self.config.effective_key_bits,
                device=self.device,
            )

    def encode_keys(
        self, keys: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Encode key vectors for cache storage.

        Args:
            keys: (batch, seq_len, num_kv_heads, head_dim)

        Returns:
            (quantized_data, metadata_dict)
        """
        meta: dict[str, Optional[torch.Tensor]] = {}

        if self.config.enable_onebit and self.onebit is not None:
            return self._encode_onebit(keys, meta)

        if self.config.enable_polar_quant and self.polar_quant_key is not None:
            return self._encode_polar(keys, self.polar_quant_key, meta, is_key=True)

        if self.asymmetric is not None:
            return self.asymmetric.quantize_keys(keys)

        # Fallback: no quantization
        return keys, meta

    def encode_values(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Encode value vectors for cache storage.

        Args:
            values: (batch, seq_len, num_kv_heads, head_dim)

        Returns:
            (quantized_data, metadata_dict)
        """
        meta: dict[str, Optional[torch.Tensor]] = {}

        if self.config.enable_onebit and self.onebit is not None:
            return self._encode_onebit(values, meta)

        if self.config.enable_polar_quant and self.polar_quant_value is not None:
            return self._encode_polar(values, self.polar_quant_value, meta, is_key=False)

        if self.asymmetric is not None:
            return self.asymmetric.quantize_values(values)

        return values, meta

    def _encode_polar(
        self,
        x: torch.Tensor,
        polar_quant: PolarQuant,
        meta: dict,
        is_key: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Encode using PolarQuant pipeline."""
        batch, seq, heads, dim = x.shape

        # Reshape for per-head quantization
        x_flat = x.reshape(-1, dim)

        # Mixed precision: separate outlier channels
        outlier_data = None
        if self.mixed_precision is not None and is_key:
            x_flat, outlier_data = self.mixed_precision.extract_outliers(x_flat)

        # PolarQuant encode (with packing for real compression)
        encoded = polar_quant.encode(x_flat, pack=True)
        meta["magnitudes"] = encoded.magnitudes.reshape(batch, seq, heads)

        # QJL residual correction
        qjl_signs = None
        if self.qjl is not None:
            reconstructed = polar_quant.decode(encoded)
            residual = x_flat - reconstructed
            qjl_signs, residual_norms = self.qjl.encode(residual)
            meta["qjl_signs"] = qjl_signs.reshape(batch, seq, heads, -1)
            meta["qjl_residual_norms"] = residual_norms.reshape(batch, seq, heads)

        # Store outlier data
        if outlier_data is not None:
            meta["outlier_indices"] = outlier_data["indices"]
            meta["outlier_values"] = outlier_data["values"].reshape(batch, seq, heads, -1)

        # Store packed data (uint8) for real compression, with shape_info for unpacking
        if encoded.packed_data is not None:
            meta["packed_shape_info"] = encoded.shape_info
            quantized = encoded.packed_data.reshape(batch, seq, heads, -1)
        else:
            quantized = encoded.quantized_indices.reshape(batch, seq, heads, -1)
        return quantized, meta

    def _encode_onebit(
        self, x: torch.Tensor, meta: dict
    ) -> tuple[torch.Tensor, dict]:
        """Encode using 1-bit Bonsai-style quantization.

        When Hadamard is enabled, rotates vectors first to eliminate outliers.
        This redistributes large magnitudes across all channels, making the
        sign-bit quantization much more effective (TurboQuant + Bonsai hybrid).
        """
        batch, seq, heads, dim = x.shape
        x_flat = x.reshape(-1, dim)

        # Apply Hadamard rotation to eliminate outliers before sign quantization
        if self.hadamard is not None:
            if dim != self._hadamard_padded_dim:
                x_flat = torch.nn.functional.pad(x_flat, (0, self._hadamard_padded_dim - dim))
            x_flat = self.hadamard.forward(x_flat)

        packed, scales = self.onebit.quantize(x_flat)
        meta["scales"] = scales.reshape(batch, seq, heads, -1)
        meta["original_shape"] = torch.tensor([batch, seq, heads, dim])
        meta["hadamard_applied"] = self.hadamard is not None
        return packed.reshape(batch, seq, heads, -1), meta

    def decode_keys(
        self, quantized: torch.Tensor, meta: dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Decode key vectors from cache storage."""
        if self.config.enable_onebit and self.onebit is not None:
            return self._decode_onebit(quantized, meta)

        if self.config.enable_polar_quant and self.polar_quant_key is not None:
            return self._decode_polar(quantized, meta, self.polar_quant_key, is_key=True)

        if self.asymmetric is not None:
            return self.asymmetric.dequantize_keys(quantized, meta)

        return quantized

    def decode_values(
        self, quantized: torch.Tensor, meta: dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Decode value vectors from cache storage."""
        if self.config.enable_onebit and self.onebit is not None:
            return self._decode_onebit(quantized, meta)

        if self.config.enable_polar_quant and self.polar_quant_value is not None:
            return self._decode_polar(quantized, meta, self.polar_quant_value, is_key=False)

        if self.asymmetric is not None:
            return self.asymmetric.dequantize_values(quantized, meta)

        return quantized

    def _decode_polar(
        self,
        quantized: torch.Tensor,
        meta: dict,
        polar_quant: PolarQuant,
        is_key: bool,
    ) -> torch.Tensor:
        """Decode PolarQuant-encoded vectors."""
        batch, seq, heads = quantized.shape[:3]
        quantized_flat = quantized.reshape(-1, quantized.shape[-1])

        from turboquant.core.polar_quant import PolarQuantOutput
        encoded = PolarQuantOutput(
            magnitudes=meta["magnitudes"].reshape(-1).float(),
            quantized_indices=quantized_flat if "packed_shape_info" not in meta else None,
            original_dim=self.config.head_dim,
        )
        # If data was packed, set packed fields for decode
        if "packed_shape_info" in meta:
            encoded.packed_data = quantized_flat.reshape(-1)
            encoded.shape_info = meta["packed_shape_info"]
            encoded.quantized_indices = None

        reconstructed = polar_quant.decode(encoded)

        # Apply QJL correction
        if self.qjl is not None and meta.get("qjl_signs") is not None:
            qjl_signs = meta["qjl_signs"].reshape(-1, meta["qjl_signs"].shape[-1])
            # Pass residual norms for proper scaling (None = backward compat)
            residual_norms = None
            if meta.get("qjl_residual_norms") is not None:
                residual_norms = meta["qjl_residual_norms"].reshape(-1)
            correction = self.qjl.decode_correction(qjl_signs, residual_norms)
            # Match dimensions: QJL operates on padded dim, reconstructed on original
            rdim = reconstructed.shape[-1]
            cdim = correction.shape[-1]
            if cdim > rdim:
                correction = correction[..., :rdim]
            elif cdim < rdim:
                correction = torch.nn.functional.pad(correction, (0, rdim - cdim))
            reconstructed = reconstructed + correction

        # Restore outlier channels
        if self.mixed_precision is not None and is_key and meta.get("outlier_values") is not None:
            outlier_vals = meta["outlier_values"].reshape(-1, meta["outlier_values"].shape[-1])
            outlier_idx = meta["outlier_indices"]
            reconstructed = self.mixed_precision.restore_outliers(
                reconstructed, outlier_idx, outlier_vals
            )

        return reconstructed.reshape(batch, seq, heads, -1)

    def _decode_onebit(self, quantized: torch.Tensor, meta: dict) -> torch.Tensor:
        """Decode 1-bit Bonsai-style quantized vectors."""
        if "original_shape" in meta:
            orig_shape = tuple(meta["original_shape"].tolist())
            batch, seq, heads, dim = orig_shape
        else:
            batch, seq, heads = quantized.shape[:3]
            dim = self.config.head_dim
        packed_flat = quantized.reshape(-1, quantized.shape[-1])
        scales = meta["scales"].reshape(-1, meta["scales"].shape[-1])

        # Dequantize -- if Hadamard was applied, dequantize in rotated space first
        hadamard_applied = meta.get("hadamard_applied", False)
        if hadamard_applied and self.hadamard is not None:
            dequant_dim = self._hadamard_padded_dim
        else:
            dequant_dim = dim

        reconstructed = self.onebit.dequantize(packed_flat, scales, dequant_dim)

        # Inverse Hadamard to recover original space
        if hadamard_applied and self.hadamard is not None:
            reconstructed = self.hadamard.inverse(reconstructed)
            if dequant_dim != dim:
                reconstructed = reconstructed[:, :dim]

        return reconstructed.reshape(batch, seq, heads, dim)

    def compute_compression_ratio(self, original_bytes: int) -> dict[str, float]:
        """Compute theoretical compression ratio."""
        bits = self.config.effective_key_bits
        if self.config.enable_onebit:
            effective_bits = 1.125  # Q1_0_g128
        elif self.config.enable_polar_quant:
            # PolarQuant: bits per coordinate + 16 bits magnitude per vector
            effective_bits = bits + 16.0 / self.config.head_dim
        else:
            effective_bits = bits

        ratio = 16.0 / effective_bits  # vs FP16 baseline
        compressed_bytes = original_bytes / ratio

        return {
            "effective_bits_per_element": effective_bits,
            "compression_ratio": ratio,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "memory_savings_pct": (1 - 1 / ratio) * 100,
        }

    def to(self, device: str) -> TurboQuantizer:
        self.device = device
        if self.polar_quant_key:
            self.polar_quant_key = self.polar_quant_key.to(device)
        if self.polar_quant_value:
            self.polar_quant_value = self.polar_quant_value.to(device)
        if self.qjl:
            self.qjl = self.qjl.to(device)
        if self.hadamard:
            self.hadamard = self.hadamard.to(device)
        return self
