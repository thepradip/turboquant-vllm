"""Quantized KV Cache manager with residual buffer.

Implements a production-ready KV cache that:
- Keeps recent tokens in full FP16 precision (residual buffer)
- Quantizes older tokens using TurboQuant/KIVI/1-bit
- Supports streaming append (new tokens quantized on the fly)
- Tracks memory usage for profiling

Groups of tokens are quantized together and stored with their own metadata,
enabling correct dequantization regardless of the quantization strategy.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional

from turboquant.config import TurboQuantConfig


@dataclass
class QuantizedGroup:
    """A group of quantized tokens with its metadata."""
    data: torch.Tensor
    meta: dict
    num_tokens: int


@dataclass
class CacheEntry:
    """A single layer's quantized KV cache."""
    # Quantized historical cache stored as groups
    key_groups: list[QuantizedGroup] = field(default_factory=list)
    value_groups: list[QuantizedGroup] = field(default_factory=list)
    # Full-precision residual buffer (recent tokens)
    key_residual: Optional[torch.Tensor] = None
    value_residual: Optional[torch.Tensor] = None
    # Sequence tracking
    quantized_len: int = 0
    residual_len: int = 0

    @property
    def total_len(self) -> int:
        return self.quantized_len + self.residual_len


class QuantizedKVCache:
    """Multi-layer quantized KV cache with residual buffer strategy.

    Architecture:
    - Each layer has a CacheEntry with quantized groups + residual buffer
    - New tokens go to the residual buffer (FP16)
    - When residual buffer fills up, oldest tokens are quantized as a group
      and appended to the compressed cache
    - At retrieval, each group is dequantized independently and concatenated
    """

    def __init__(self, config: TurboQuantConfig, num_layers: int):
        self.config = config
        self.num_layers = num_layers
        self.entries: list[CacheEntry] = [CacheEntry() for _ in range(num_layers)]
        self._quantizer = None  # Set by TurboQuantizer

    def set_quantizer(self, quantizer) -> None:
        """Set the quantizer used for compressing cache entries."""
        self._quantizer = quantizer

    def append(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Append new KV pairs to a layer's cache.

        Args:
            layer_idx: Which transformer layer
            keys: (batch, new_tokens, num_kv_heads, head_dim)
            values: (batch, new_tokens, num_kv_heads, head_dim)
        """
        entry = self.entries[layer_idx]

        # Initialize residual buffers if needed
        if entry.key_residual is None:
            entry.key_residual = keys
            entry.value_residual = values
            entry.residual_len = keys.shape[1]
        else:
            entry.key_residual = torch.cat([entry.key_residual, keys], dim=1)
            entry.value_residual = torch.cat([entry.value_residual, values], dim=1)
            entry.residual_len = entry.key_residual.shape[1]

        # Flush to quantized cache if residual exceeds buffer size
        while entry.residual_len > self.config.residual_buffer_size:
            self._flush_residual(layer_idx)

    def _flush_residual(self, layer_idx: int) -> None:
        """Move oldest tokens from residual buffer to quantized cache."""
        entry = self.entries[layer_idx]
        if self._quantizer is None or entry.key_residual is None:
            return

        group_size = self.config.kivi_group_size
        if entry.residual_len < group_size:
            return

        # Take the oldest group_size tokens for quantization
        keys_to_quantize = entry.key_residual[:, :group_size]
        values_to_quantize = entry.value_residual[:, :group_size]

        # Quantize
        k_quant, k_meta = self._quantizer.encode_keys(keys_to_quantize)
        v_quant, v_meta = self._quantizer.encode_values(values_to_quantize)

        # Store as groups with their own metadata
        entry.key_groups.append(QuantizedGroup(
            data=k_quant, meta=k_meta, num_tokens=group_size
        ))
        entry.value_groups.append(QuantizedGroup(
            data=v_quant, meta=v_meta, num_tokens=group_size
        ))

        entry.quantized_len += group_size

        # Remove flushed tokens from residual
        entry.key_residual = entry.key_residual[:, group_size:]
        entry.value_residual = entry.value_residual[:, group_size:]
        entry.residual_len = entry.key_residual.shape[1]

    def get_keys(self, layer_idx: int) -> torch.Tensor:
        """Get full key cache for attention (dequantized + residual)."""
        entry = self.entries[layer_idx]
        parts = []

        # Dequantize each group independently
        for group in entry.key_groups:
            dequantized = self._quantizer.decode_keys(group.data, group.meta)
            parts.append(dequantized)

        if entry.key_residual is not None and entry.residual_len > 0:
            parts.append(entry.key_residual)

        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=1)

    def get_values(self, layer_idx: int) -> torch.Tensor:
        """Get full value cache for attention (dequantized + residual)."""
        entry = self.entries[layer_idx]
        parts = []

        for group in entry.value_groups:
            dequantized = self._quantizer.decode_values(group.data, group.meta)
            parts.append(dequantized)

        if entry.value_residual is not None and entry.residual_len > 0:
            parts.append(entry.value_residual)

        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=1)

    def get_seq_len(self, layer_idx: int) -> int:
        return self.entries[layer_idx].total_len

    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries = [CacheEntry() for _ in range(self.num_layers)]

    def memory_usage(self) -> dict[str, float]:
        """Compute memory usage in bytes."""
        quantized_bytes = 0
        residual_bytes = 0
        metadata_bytes = 0

        for entry in self.entries:
            for groups in [entry.key_groups, entry.value_groups]:
                for group in groups:
                    quantized_bytes += group.data.nelement() * group.data.element_size()
                    for v in group.meta.values():
                        if isinstance(v, torch.Tensor):
                            metadata_bytes += v.nelement() * v.element_size()

            if entry.key_residual is not None:
                residual_bytes += entry.key_residual.nelement() * entry.key_residual.element_size()
            if entry.value_residual is not None:
                residual_bytes += entry.value_residual.nelement() * entry.value_residual.element_size()

        total = quantized_bytes + residual_bytes + metadata_bytes
        return {
            "quantized_bytes": quantized_bytes,
            "residual_bytes": residual_bytes,
            "metadata_bytes": metadata_bytes,
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
        }
