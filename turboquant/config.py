"""Configuration for TurboQuant KV cache quantization."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TurboQuantConfig:
    """Configuration for the TurboQuant KV cache quantization system.

    Supports multiple quantization strategies:
    - TurboQuant (PolarQuant + QJL): calibration-free online vector quantization
    - KIVI: asymmetric per-channel keys / per-token values
    - Mixed precision: outlier channels at higher bit-width
    - 1-bit (Bonsai-inspired): Q1_0_g128 for extreme compression
    """

    # -- Bit-width configuration --
    kv_bits: int = 4
    key_bits: Optional[int] = None   # Override for keys (asymmetric)
    value_bits: Optional[int] = None  # Override for values (asymmetric)

    # -- PolarQuant settings --
    enable_polar_quant: bool = True
    polar_quant_bits: int = 2  # Bits allocated to PolarQuant stage
    codebook_size: int = 16  # Lloyd-Max codebook entries (2^bits)

    # -- QJL residual correction --
    enable_qjl: bool = False  # Disabled by default (community finding)
    qjl_projection_dim: int = 256
    qjl_bits: int = 1  # Sign-bit quantization

    # -- Hadamard rotation --
    enable_hadamard: bool = True
    hadamard_block_size: int = 128  # Must be power of 2

    # -- KIVI asymmetric quantization --
    enable_asymmetric: bool = True
    key_quant_mode: str = "per_channel"
    value_quant_mode: str = "per_token"
    kivi_group_size: int = 32  # Tokens per quantization group

    # -- Residual buffer --
    residual_buffer_size: int = 128  # Recent tokens kept in FP16

    # -- Mixed precision --
    enable_mixed_precision: bool = True
    outlier_channel_fraction: float = 0.25  # Top 25% channels at higher bits
    outlier_bits: int = 8  # Bits for outlier channels

    # -- 1-bit quantization (Bonsai-inspired) --
    enable_onebit: bool = False
    onebit_group_size: int = 128  # Q1_0_g128 format

    # -- Model dimensions --
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 65536

    # -- Runtime --
    device: str = "cpu"
    dtype: str = "float16"

    @property
    def effective_key_bits(self) -> int:
        return self.key_bits if self.key_bits is not None else self.kv_bits

    @property
    def effective_value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.kv_bits

    @classmethod
    def from_yaml(cls, path: str | Path) -> TurboQuantConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def turbo_4bit(cls, **kwargs) -> TurboQuantConfig:
        """Preset: TurboQuant 4-bit (PolarQuant only, no QJL)."""
        return cls(
            kv_bits=4, key_bits=4, value_bits=4,
            enable_polar_quant=True, polar_quant_bits=4,
            enable_qjl=False,
            enable_hadamard=True,
            enable_asymmetric=True,
            enable_mixed_precision=True,
            **kwargs,
        )

    @classmethod
    def kivi_2bit(cls, **kwargs) -> TurboQuantConfig:
        """Preset: KIVI-style asymmetric 2-bit."""
        return cls(
            kv_bits=2, key_bits=4, value_bits=2,
            enable_polar_quant=False,
            enable_qjl=False,
            enable_hadamard=True,
            enable_asymmetric=True,
            enable_mixed_precision=False,
            **kwargs,
        )

    @classmethod
    def turbo_3bit(cls, **kwargs) -> TurboQuantConfig:
        """Preset: TurboQuant 3-bit (2-bit PolarQuant + 1-bit QJL)."""
        return cls(
            kv_bits=3,
            enable_polar_quant=True, polar_quant_bits=2,
            enable_qjl=True, qjl_bits=1,
            enable_hadamard=True,
            enable_asymmetric=True,
            enable_mixed_precision=True,
            outlier_channel_fraction=0.25,
            **kwargs,
        )

    @classmethod
    def onebit_extreme(cls, **kwargs) -> TurboQuantConfig:
        """Preset: 1-bit Bonsai-inspired extreme compression."""
        return cls(
            kv_bits=1,
            enable_polar_quant=False,
            enable_qjl=False,
            enable_hadamard=True,
            enable_asymmetric=False,
            enable_onebit=True,
            onebit_group_size=128,
            enable_mixed_precision=False,
            **kwargs,
        )
