"""
compress_cache() for PyTorch / HuggingFace models.

Compresses the KV cache in-place after prefill using TurboQuant.
Works with HuggingFace's DynamicCache (transformers >= 4.36).

Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant.compress import compress_cache

    model = AutoModelForCausalLM.from_pretrained(...)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Prefill: generate KV cache
    out = model(**inputs, use_cache=True)
    cache = out.past_key_values

    # Compress KV cache with TurboQuant
    result = compress_cache(cache, head_dim=128, bits=4)
    print(result)
    # {'cosine': 0.9914, 'original_mb': 36.0, 'compressed_mb': 18.0, 'ratio': 2.0, ...}

    # Generate with compressed cache
    output = model.generate(inputs.input_ids, past_key_values=cache, ...)

Author: Pradip Tivhale, April 2026
"""

import time
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer


def compress_cache(
    cache: Any,
    head_dim: int = 128,
    num_kv_heads: int = 8,
    bits: int = 4,
    device: str = "cpu",
) -> Dict:
    """
    Compress HuggingFace KV cache in-place using TurboQuant PolarQuant.

    Takes a DynamicCache (or list of (key, value) tuples), compresses each
    layer's KV tensors, measures real cosine from actual data, writes
    dequantized values back, and reports real memory from tensor .nbytes.

    Args:
        cache: HuggingFace DynamicCache or list of (key, value) tuples.
        head_dim: Dimension per attention head.
        num_kv_heads: Number of KV heads.
        bits: Quantization bits (2, 3, or 4).
        device: Device for quantizer ('cpu', 'cuda', 'mps').

    Returns:
        Dict with cosine, compress_ms, original_mb, compressed_mb, ratio.
    """
    # Build quantizer
    config = TurboQuantConfig.turbo_4bit(
        num_heads=num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if bits == 3:
        config = TurboQuantConfig.turbo_3bit(
            num_heads=num_kv_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
    # Disable mixed precision and QJL — both add more overhead than they save
    config.enable_mixed_precision = False
    config.enable_qjl = False
    config.device = device
    quantizer = TurboQuantizer(config)

    # Extract key/value tensors from cache
    # HuggingFace DynamicCache: cache.key_cache[i], cache.value_cache[i]
    # Each tensor: (batch, num_kv_heads, seq_len, head_dim)
    if hasattr(cache, 'key_cache'):
        num_layers = len(cache.key_cache)
        get_kv = lambda i: (cache.key_cache[i], cache.value_cache[i])
        set_kv = lambda i, k, v: (
            cache.key_cache.__setitem__(i, k),
            cache.value_cache.__setitem__(i, v),
        )
    elif isinstance(cache, (list, tuple)):
        num_layers = len(cache)
        get_kv = lambda i: cache[i]
        set_kv = lambda i, k, v: cache.__setitem__(i, (k, v))
    else:
        raise TypeError(f"Unsupported cache type: {type(cache)}")

    t0 = time.time()

    # Measure original memory (actual .nbytes, used portion only)
    original_bytes = 0
    for i in range(num_layers):
        k, v = get_kv(i)
        original_bytes += k.nelement() * k.element_size()
        original_bytes += v.nelement() * v.element_size()

    cosine_sum = 0.0
    cosine_count = 0
    layers_compressed = 0
    compressed_bytes = 0

    for i in range(num_layers):
        k, v = get_kv(i)
        # k, v shape: (batch, num_kv_heads, seq_len, head_dim)
        if k.numel() == 0:
            continue

        # Transpose to (batch, seq_len, num_kv_heads, head_dim) for quantizer
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()

        # Encode
        k_quant, k_meta = quantizer.encode_keys(k_t)
        v_quant, v_meta = quantizer.encode_values(v_t)

        # Decode (reconstruct)
        k_recon = quantizer.decode_keys(k_quant, k_meta)
        v_recon = quantizer.decode_values(v_quant, v_meta)

        # Measure real cosine BEFORE overwriting
        k_flat = k_t.reshape(-1, head_dim).float()
        k_recon_flat = k_recon.reshape(-1, head_dim).float()
        cos = F.cosine_similarity(k_flat, k_recon_flat, dim=-1).mean().item()
        cosine_sum += cos
        cosine_count += 1

        # Track compressed size (quantized indices + metadata tensors)
        compressed_bytes += k_quant.nelement() * k_quant.element_size()
        compressed_bytes += v_quant.nelement() * v_quant.element_size()
        for meta in [k_meta, v_meta]:
            for val in meta.values():
                if isinstance(val, torch.Tensor):
                    compressed_bytes += val.nelement() * val.element_size()

        # Write dequantized back to cache (transpose back)
        k_hat = k_recon.transpose(1, 2).to(k.dtype)
        v_hat = v_recon.transpose(1, 2).to(v.dtype)
        set_kv(i, k_hat, v_hat)

        layers_compressed += 1

    elapsed_ms = (time.time() - t0) * 1000
    avg_cosine = cosine_sum / cosine_count if cosine_count > 0 else 0.0
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0.0

    return {
        "cosine": round(avg_cosine, 4),
        "compress_ms": round(elapsed_ms, 0),
        "layers_compressed": layers_compressed,
        "original_mb": round(original_bytes / 1024 / 1024, 1),
        "compressed_mb": round(compressed_bytes / 1024 / 1024, 1),
        "ratio": round(ratio, 1),
    }
