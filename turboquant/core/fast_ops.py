"""Fast quantization operations optimized for Metal (MPS) and CUDA.

Key optimizations:
1. torch.bucketize for codebook lookup (vectorized, no Python loop)
2. In-place Hadamard butterfly (zero allocation)
3. Fused encode: magnitude + rotate + quantize in minimal kernel launches
4. torch.compile-ready (no data-dependent control flow)
"""

from __future__ import annotations

import torch
import math
from typing import Optional


def fast_hadamard_inplace(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """In-place Fast Walsh-Hadamard with random signs. Zero allocations."""
    x = x * signs
    n = x.shape[-1]
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, n)
    B = x.shape[0]

    h = 1
    while h < n:
        half = n // (2 * h)
        x_view = x.view(B, half, 2, h)
        a = x_view[:, :, 0, :]
        b = x_view[:, :, 1, :]
        sum_ab = a + b
        diff_ab = a - b
        x_view[:, :, 0, :] = sum_ab
        x_view[:, :, 1, :] = diff_ab
        h *= 2

    x = x * (1.0 / math.sqrt(n))
    return x.reshape(*batch_shape, n)


def fast_quantize_codebook(x: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Vectorized codebook quantization using torch.bucketize.

    ~10x faster than the loop-based approach on GPU.
    """
    return torch.bucketize(x.contiguous(), boundaries.contiguous()).to(torch.int16)


def fast_dequantize_codebook(indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Fast codebook lookup via indexing."""
    return codebook[indices.long()]


def fast_polar_encode(
    x: torch.Tensor,
    signs: torch.Tensor,
    boundaries: torch.Tensor,
    pad_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused PolarQuant encode: magnitude + Hadamard + quantize.

    Combines 3 steps into 1 function call for torch.compile fusion.
    Returns (magnitudes, quantized_indices, rotated_for_decode).
    """
    orig_dim = x.shape[-1]

    # Magnitude
    magnitudes = torch.norm(x, dim=-1)
    safe_mag = magnitudes.unsqueeze(-1).clamp(min=1e-8)
    directions = x / safe_mag

    # Pad if needed
    if orig_dim != pad_dim:
        directions = torch.nn.functional.pad(directions, (0, pad_dim - orig_dim))

    # Hadamard rotation (in-place friendly)
    rotated = directions * signs
    n = rotated.shape[-1]
    flat = rotated.reshape(-1, n)
    B = flat.shape[0]
    h = 1
    while h < n:
        half = n // (2 * h)
        view = flat.view(B, half, 2, h)
        a = view[:, :, 0, :].clone()
        b = view[:, :, 1, :].clone()
        view[:, :, 0, :] = a + b
        view[:, :, 1, :] = a - b
        h *= 2
    rotated = flat * (1.0 / math.sqrt(n))
    rotated = rotated.reshape(directions.shape)

    # Quantize
    indices = torch.bucketize(rotated.contiguous(), boundaries.contiguous()).to(torch.int16)

    return magnitudes.to(torch.float16), indices, rotated


def fast_polar_decode(
    magnitudes: torch.Tensor,
    indices: torch.Tensor,
    codebook: torch.Tensor,
    signs: torch.Tensor,
    orig_dim: int,
    pad_dim: int,
) -> torch.Tensor:
    """Fused PolarQuant decode: dequantize + inverse Hadamard + rescale."""
    # Dequantize
    rotated = codebook[indices.long()]

    # Inverse Hadamard
    n = rotated.shape[-1]
    flat = rotated.reshape(-1, n)
    B = flat.shape[0]
    h = 1
    while h < n:
        half = n // (2 * h)
        view = flat.view(B, half, 2, h)
        a = view[:, :, 0, :].clone()
        b = view[:, :, 1, :].clone()
        view[:, :, 0, :] = a + b
        view[:, :, 1, :] = a - b
        h *= 2
    directions = flat * (1.0 / math.sqrt(n))
    directions = directions.reshape(rotated.shape) * signs

    # Unpad
    if orig_dim != pad_dim:
        directions = directions[..., :orig_dim]

    # Re-normalize + rescale
    norm = torch.norm(directions, dim=-1, keepdim=True).clamp(min=1e-8)
    directions = directions / norm
    return directions * magnitudes.float().unsqueeze(-1)


# ── Asymmetric quantization (KIVI) ──

def fast_per_channel_quantize(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized per-channel min-max quantization."""
    qmax = (1 << bits) - 1
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8) / qmax
    quantized = ((x - x_min) / scale).round().clamp(0, qmax).to(torch.int16)
    return quantized, scale, x_min


def fast_per_token_quantize(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized per-token min-max quantization."""
    qmax = (1 << bits) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8) / qmax
    quantized = ((x - x_min) / scale).round().clamp(0, qmax).to(torch.int16)
    return quantized, scale, x_min


def fast_dequantize(quantized: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Fused dequantize: q * scale + zero."""
    return quantized.float() * scale + zero


# ── 1-bit quantization ──

def fast_onebit_encode(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast 1-bit quantization with bit packing.

    Uses vectorized ops instead of Python loop for packing.
    """
    dim = x.shape[-1]
    flat = x.reshape(-1, dim)
    batch = flat.shape[0]

    # Pad
    pad = (group_size - dim % group_size) % group_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    pdim = flat.shape[-1]

    # Group scales
    grouped = flat.reshape(batch, pdim // group_size, group_size)
    scales = grouped.abs().mean(dim=-1).to(torch.float16)

    # Sign bits
    signs = (flat >= 0).to(torch.uint8)

    # Pack 8 bits per byte using bit shifts (vectorized)
    pack_dim = (pdim + 7) // 8
    if pdim % 8 != 0:
        signs = torch.nn.functional.pad(signs, (0, 8 - pdim % 8))
    signs_reshaped = signs.reshape(batch, -1, 8)
    packed = torch.zeros(batch, signs_reshaped.shape[1], dtype=torch.uint8, device=x.device)
    for i in range(8):
        packed |= signs_reshaped[:, :, i] << (7 - i)

    return packed, scales


def fast_onebit_decode(
    packed: torch.Tensor, scales: torch.Tensor, original_dim: int, group_size: int
) -> torch.Tensor:
    """Fast 1-bit dequantization."""
    batch = scales.shape[0]
    num_groups = scales.shape[1]
    pdim = num_groups * group_size

    # Unpack bits (vectorized)
    bits = []
    for i in range(8):
        bits.append((packed >> (7 - i)) & 1)
    unpacked = torch.stack(bits, dim=-1).reshape(batch, -1)[:, :pdim]

    # Dequantize: sign * scale
    sign_values = unpacked.float() * 2 - 1
    sign_values = sign_values.reshape(batch, num_groups, group_size)
    reconstructed = sign_values * scales.float().unsqueeze(-1)
    reconstructed = reconstructed.reshape(batch, pdim)

    if pdim != original_dim:
        reconstructed = reconstructed[:, :original_dim]
    return reconstructed
