"""Quantized attention computation.

Computes attention over the quantized KV cache by:
1. Dequantizing keys and values from the compressed cache
2. Computing standard scaled dot-product attention
3. Supporting both the quantized (historical) and full-precision (residual) portions

For production, a fused decode+attention kernel would avoid materializing
the full FP16 cache, but this reference implementation prioritizes correctness
and testability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math
from typing import Optional

from turboquant.core.kv_cache import QuantizedKVCache


class QuantizedAttention:
    """Scaled dot-product attention with quantized KV cache support."""

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dropout: float = 0.0,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(head_dim)

        # GQA ratio
        assert num_heads % num_kv_heads == 0
        self.num_groups = num_heads // num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: QuantizedKVCache,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention with quantized KV cache.

        Args:
            query: (batch, seq_len, num_heads, head_dim) -- current query
            kv_cache: The quantized KV cache
            layer_idx: Which transformer layer
            attention_mask: Optional causal mask

        Returns:
            Attention output: (batch, seq_len, num_heads, head_dim)
        """
        # Get full keys and values (dequantized + residual)
        keys = kv_cache.get_keys(layer_idx)    # (batch, kv_len, num_kv_heads, head_dim)
        values = kv_cache.get_values(layer_idx)  # (batch, kv_len, num_kv_heads, head_dim)

        if isinstance(keys, torch.Tensor) and keys.numel() == 0:
            return torch.zeros_like(query)

        return self.compute_attention(query, keys, values, attention_mask)

    def compute_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with GQA support.

        Args:
            query: (batch, q_len, num_heads, head_dim)
            keys: (batch, kv_len, num_kv_heads, head_dim)
            values: (batch, kv_len, num_kv_heads, head_dim)
            attention_mask: (batch, 1, q_len, kv_len) or broadcastable

        Returns:
            (batch, q_len, num_heads, head_dim)
        """
        batch = query.shape[0]
        q_len = query.shape[1]
        kv_len = keys.shape[1]

        # Transpose to (batch, heads, seq, dim) for bmm
        q = query.transpose(1, 2)  # (batch, num_heads, q_len, head_dim)
        k = keys.transpose(1, 2)   # (batch, num_kv_heads, kv_len, head_dim)
        v = values.transpose(1, 2)  # (batch, num_kv_heads, kv_len, head_dim)

        # Expand KV heads for GQA
        if self.num_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            k = k.reshape(batch, self.num_heads, kv_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            v = v.reshape(batch, self.num_heads, kv_len, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        output = torch.matmul(attn_weights, v)  # (batch, num_heads, q_len, head_dim)

        # Transpose back
        return output.transpose(1, 2)  # (batch, q_len, num_heads, head_dim)

    def compute_attention_error(
        self,
        query: torch.Tensor,
        original_keys: torch.Tensor,
        original_values: torch.Tensor,
        quantized_keys: torch.Tensor,
        quantized_values: torch.Tensor,
    ) -> dict[str, float]:
        """Compare attention output between original and quantized KV.

        This is the gold-standard quality metric: the actual impact of
        quantization on attention output, not just individual vector error.
        """
        original_output = self.compute_attention(query, original_keys, original_values)
        quantized_output = self.compute_attention(query, quantized_keys, quantized_values)

        mse = ((original_output - quantized_output) ** 2).mean().item()
        cos_sim = F.cosine_similarity(
            original_output.reshape(-1, self.head_dim),
            quantized_output.reshape(-1, self.head_dim),
            dim=-1,
        )
        relative_error = (
            torch.norm(original_output - quantized_output)
            / torch.norm(original_output).clamp(min=1e-8)
        ).item()

        return {
            "attention_mse": mse,
            "attention_cosine_similarity_mean": cos_sim.mean().item(),
            "attention_cosine_similarity_min": cos_sim.min().item(),
            "attention_relative_error": relative_error,
        }
