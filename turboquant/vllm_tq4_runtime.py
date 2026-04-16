"""Runtime hooks for packed TurboQuant 4-bit KV cache in vLLM.

This is a proof-oriented integration path: KV pages are stored in a packed
4-bit PolarQuant format, then decoded to an fp16 scratch cache before calling
vLLM's existing Triton attention kernel. It is expected to save KV cache
allocation memory, but it is not the final fused-kernel TurboQuant path.
"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import replace
from math import lcm
import os
from typing import Any

import torch

from turboquant.core.fast_ops import fast_polar_decode, fast_polar_encode
from turboquant.core.lloyd_max import LloydMaxQuantizer

try:
    from vllm.triton_utils import tl, triton
except Exception:  # pragma: no cover - vLLM may not be installed.
    tl = None
    triton = None


META_BYTES = 2  # fp16 magnitude per token/head
TQ4_DTYPES = {"turboquant", "turboquant_4bit"}
_FUSED_HEAD_SIZES = frozenset({64, 128, 256})
_COMPILE_ENV = "TURBOQUANT_COMPILE_POLAR"
_FUSED_DECODE_ENV = "TURBOQUANT_FUSED_DECODE"
_FUSED_DEBUG_ENV = "TURBOQUANT_DEBUG_FUSED"
_PATH_DEBUG_ENV = "TURBOQUANT_DEBUG_PATHS"
_COMPARE_ENV = "TURBOQUANT_DEBUG_COMPARE"
_DECODE_CACHE_STATE: dict[tuple[int, torch.device, torch.dtype, int], dict[str, Any]] = {}


def _debug_paths_enabled() -> bool:
    return os.environ.get(_PATH_DEBUG_ENV, "0") == "1"


def _debug_log_path(message: str) -> None:
    if _debug_paths_enabled():
        print(f"TurboQuant path: {message}", flush=True)


def _debug_compare_enabled() -> bool:
    return os.environ.get(_COMPARE_ENV, "0") == "1"


def _summarize_compare(tag: str, original: torch.Tensor, decoded: torch.Tensor) -> None:
    if not _debug_compare_enabled():
        return
    orig = original.reshape(-1, original.shape[-1]).float()
    recon = decoded.reshape(-1, decoded.shape[-1]).float()
    mse = torch.mean((orig - recon) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(orig, recon, dim=-1).mean().item()
    print(
        f"TurboQuant compare: {tag} mse={mse:.6f} cos={cos:.6f} "
        f"shape={tuple(original.shape)}",
        flush=True,
    )


def is_tq4_dtype(dtype: str) -> bool:
    return dtype in TQ4_DTYPES


def packed_tq4_width(head_size: int) -> int:
    return (head_size + 1) // 2 + META_BYTES


def tq4_block_size(base_block_size: int, num_kv_heads: int, head_size: int, head_size_v: int) -> int:
    """Scale block size so packed pages align with vLLM hybrid cache pages.

    Qwen3.5 uses hybrid attention/Mamba cache management. vLLM's model config
    pass chooses an attention block size using the logical cache dtype and then
    pads Mamba pages to that size. Our actual packed page has fewer bytes per
    token, so we increase the TQ4 attention block size to keep page bytes
    compatible while reducing page count.
    """
    packed_width = packed_tq4_width(head_size)
    fp_bytes_per_token = num_kv_heads * (head_size + head_size_v) * 2
    tq_bytes_per_token = num_kv_heads * (packed_width + packed_width)
    target = base_block_size * fp_bytes_per_token
    if target % tq_bytes_per_token == 0:
        return target // tq_bytes_per_token
    # Fall back to the next 16-aligned block size and let page padding handle
    # odd models rather than silently returning an incompatible page.
    return ((target + tq_bytes_per_token - 1) // tq_bytes_per_token + 15) // 16 * 16


@lru_cache(maxsize=64)
def _cpu_codebook_and_boundaries(head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    padded_dim = 1 << (head_size - 1).bit_length() if head_size & (head_size - 1) else head_size
    lm = LloydMaxQuantizer(bits=4, dim=padded_dim)
    return lm.codebook.cpu(), lm.boundaries.cpu()


@lru_cache(maxsize=64)
def _cpu_signs(head_size: int, seed: int) -> torch.Tensor:
    padded_dim = 1 << (head_size - 1).bit_length() if head_size & (head_size - 1) else head_size
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return ((torch.randint(0, 2, (padded_dim,), generator=gen) * 2 - 1).float()).cpu()


@lru_cache(maxsize=128)
def _device_params(
    head_size: int,
    device_type: str,
    device_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = torch.device(device_type) if device_index < 0 else torch.device(device_type, device_index)
    padded_dim = 1 << (head_size - 1).bit_length() if head_size & (head_size - 1) else head_size
    codebook, boundaries = _cpu_codebook_and_boundaries(head_size)
    signs = _cpu_signs(head_size, seed)
    return (
        codebook.to(device=device),
        boundaries.to(device=device),
        signs.to(device=device),
        padded_dim,
    )


def _params(head_size: int, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    index = -1 if device.index is None else device.index
    return _device_params(head_size, device.type, index, seed)


def _use_compiled_polar(x: torch.Tensor) -> bool:
    return x.is_cuda and os.environ.get(_COMPILE_ENV, "0") == "1"


@lru_cache(maxsize=1)
def _compiled_polar_encode() -> Any:
    return torch.compile(fast_polar_encode, mode="reduce-overhead")


@lru_cache(maxsize=1)
def _compiled_polar_decode() -> Any:
    return torch.compile(fast_polar_decode, mode="reduce-overhead")


def _polar_encode(
    x: torch.Tensor,
    signs: torch.Tensor,
    boundaries: torch.Tensor,
    padded_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _use_compiled_polar(x):
        try:
            return _compiled_polar_encode()(x, signs, boundaries, padded_dim)
        except Exception:
            os.environ[_COMPILE_ENV] = "0"
    return fast_polar_encode(x, signs, boundaries, padded_dim)


def _polar_decode(
    magnitudes: torch.Tensor,
    indices: torch.Tensor,
    codebook: torch.Tensor,
    signs: torch.Tensor,
    head_size: int,
    padded_dim: int,
) -> torch.Tensor:
    if _use_compiled_polar(indices):
        try:
            return _compiled_polar_decode()(
                magnitudes,
                indices,
                codebook,
                signs,
                head_size,
                padded_dim,
            )
        except Exception:
            os.environ[_COMPILE_ENV] = "0"
    return fast_polar_decode(magnitudes, indices, codebook, signs, head_size, padded_dim)


def _pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    q = indices.to(torch.uint8).contiguous()
    if q.shape[-1] % 2:
        q = torch.nn.functional.pad(q, (0, 1))
    q2 = q.reshape(*q.shape[:-1], q.shape[-1] // 2, 2)
    return (q2[..., 0] | (q2[..., 1] << 4)).contiguous()


def _unpack_nibbles(packed: torch.Tensor, head_size: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out[..., :head_size].to(torch.int16)


def _magnitude_to_bytes(magnitudes: torch.Tensor) -> torch.Tensor:
    return magnitudes.to(torch.float16).contiguous().unsqueeze(-1).view(torch.uint8)


def _bytes_to_magnitude(raw: torch.Tensor) -> torch.Tensor:
    return raw.contiguous().view(torch.float16).squeeze(-1)


def _supports_fused_tq4_decode(head_size: int, head_size_v: int) -> bool:
    return head_size in _FUSED_HEAD_SIZES and head_size_v == head_size


def _decode_cache_key(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    dtype: torch.dtype,
) -> tuple[int, torch.device, torch.dtype, int]:
    return (kv_cache.data_ptr(), kv_cache.device, dtype, head_size)


def _get_decode_cache_state(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    key = _decode_cache_key(kv_cache, head_size=head_size, dtype=dtype)
    state = _DECODE_CACHE_STATE.get(key)
    expected_shape = tuple(kv_cache.shape)
    if state is None or state["kv_shape"] != expected_shape:
        num_blocks = kv_cache.shape[0]
        state = {
            "kv_shape": expected_shape,
            "block_versions": torch.zeros(num_blocks, dtype=torch.int64, device=kv_cache.device),
            "decoded_versions": torch.full((num_blocks,), -1, dtype=torch.int64, device=kv_cache.device),
            "decoded_cache": torch.empty(
                num_blocks,
                2,
                kv_cache.shape[2],
                kv_cache.shape[3],
                head_size,
                dtype=dtype,
                device=kv_cache.device,
            ),
        }
        _DECODE_CACHE_STATE[key] = state
    return state


def _mark_tq4_blocks_dirty(kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    if slot_mapping.numel() == 0:
        return
    block_size = kv_cache.shape[2]
    block_ids = torch.unique(slot_mapping.to(torch.long) // block_size)
    if block_ids.numel() == 0:
        return
    for cache_key, state in list(_DECODE_CACHE_STATE.items()):
        cache_ptr, cache_device, _, _ = cache_key
        if cache_ptr != kv_cache.data_ptr() or cache_device != kv_cache.device:
            continue
        state["block_versions"][block_ids] += 1


if triton is not None:

    @triton.jit
    def _hadamard_sign(rows, cols):
        parity = (
            ((rows & 1) * (cols & 1))
            + (((rows >> 1) & 1) * ((cols >> 1) & 1))
            + (((rows >> 2) & 1) * ((cols >> 2) & 1))
            + (((rows >> 3) & 1) * ((cols >> 3) & 1))
            + (((rows >> 4) & 1) * ((cols >> 4) & 1))
            + (((rows >> 5) & 1) * ((cols >> 5) & 1))
            + (((rows >> 6) & 1) * ((cols >> 6) & 1))
            + (((rows >> 7) & 1) * ((cols >> 7) & 1))
        ) & 1
        return 1.0 - 2.0 * parity.to(tl.float32)

    @triton.jit
    def _tanh(x):
        return 2 * tl.sigmoid(2 * x) - 1

    @triton.jit
    def _load_tq4_tile(
        cache,
        block_table,
        codebook,
        signs,
        seq_idx,
        kv_head_idx,
        seq_offsets,
        dims,
        packed_dims,
        block_table_stride: tl.constexpr,
        stride_cache_0: tl.constexpr,
        stride_cache_1: tl.constexpr,
        stride_cache_2: tl.constexpr,
        stride_cache_3: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        INV_SQRT_HEAD: tl.constexpr,
    ):
        physical_block_idx = tl.load(
            block_table + seq_idx * block_table_stride + seq_offsets // BLOCK_SIZE
        ).to(tl.int64)
        block_offsets = seq_offsets % BLOCK_SIZE
        byte_offsets = packed_dims // 2

        packed = tl.load(
            cache
            + physical_block_idx[None, :] * stride_cache_0
            + block_offsets[None, :] * stride_cache_1
            + kv_head_idx * stride_cache_2
            + byte_offsets[:, None] * stride_cache_3,
            mask=dims[:, None] < HEAD_SIZE,
            other=0,
        ).to(tl.uint8)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        qidx = tl.where((packed_dims[:, None] & 1) == 0, low, high)
        rotated = tl.load(codebook + qidx.to(tl.int64)).to(tl.float32)

        norm = tl.sqrt(tl.sum(rotated * rotated, axis=0))
        norm = tl.maximum(norm, 1.0e-8)

        packed_width = (HEAD_SIZE + 1) // 2
        mag_lo = tl.load(
            cache
            + physical_block_idx * stride_cache_0
            + block_offsets * stride_cache_1
            + kv_head_idx * stride_cache_2
            + (packed_width + 0) * stride_cache_3
        ).to(tl.uint16)
        mag_hi = tl.load(
            cache
            + physical_block_idx * stride_cache_0
            + block_offsets * stride_cache_1
            + kv_head_idx * stride_cache_2
            + (packed_width + 1) * stride_cache_3
        ).to(tl.uint16)
        mag = (mag_lo | (mag_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        h = _hadamard_sign(dims[:, None], packed_dims[None, :])
        decoded = tl.dot(h, rotated) * INV_SQRT_HEAD
        decoded = decoded * tl.load(signs + dims).to(tl.float32)[:, None]
        decoded = decoded * (mag / norm)[None, :]
        return decoded

    @triton.jit
    def _tq4_decode_attention_kernel(
        output,
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        key_codebook,
        key_signs,
        value_codebook,
        value_signs,
        softmax_scale: tl.constexpr,
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        block_table_stride: tl.constexpr,
        query_stride_0: tl.constexpr,
        query_stride_1: tl.constexpr,
        output_stride_0: tl.constexpr,
        output_stride_1: tl.constexpr,
        stride_k_cache_0: tl.constexpr,
        stride_k_cache_1: tl.constexpr,
        stride_k_cache_2: tl.constexpr,
        stride_k_cache_3: tl.constexpr,
        stride_v_cache_0: tl.constexpr,
        stride_v_cache_1: tl.constexpr,
        stride_v_cache_2: tl.constexpr,
        stride_v_cache_3: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        INV_SQRT_HEAD: tl.constexpr,
        LOGIT_CAP: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        query_head_idx = tl.program_id(1)
        kv_head_idx = query_head_idx // num_queries_per_kv

        dims = tl.arange(0, HEAD_SIZE)
        tile = tl.arange(0, TILE_SIZE)

        query_idx = tl.load(query_start_loc + seq_idx)
        q = tl.load(
            query
            + query_idx * query_stride_0
            + query_head_idx * query_stride_1
            + dims,
        ).to(tl.float32)
        q_signed = q * tl.load(key_signs + dims).to(tl.float32)
        h_q = _hadamard_sign(dims[:, None], dims[None, :])
        q_rot = tl.sum(h_q * q_signed[None, :], axis=1) * INV_SQRT_HEAD

        seq_len = tl.load(seq_lens + seq_idx)
        start_pos = 0
        if SLIDING_WINDOW > 0:
            start_pos = tl.maximum(0, seq_len - SLIDING_WINDOW)

        m_i = tl.full((), float("-inf"), dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)
        acc_rot = tl.zeros((HEAD_SIZE,), dtype=tl.float32)

        for tile_start in range(start_pos, seq_len, TILE_SIZE):
            seq_offsets = tile_start + tile
            tile_mask = seq_offsets < seq_len

            k_rot = _load_tq4_rotated_chunk(
                key_cache,
                block_table,
                key_codebook,
                seq_idx,
                kv_head_idx,
                seq_offsets,
                dims,
                block_table_stride,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                HEAD_SIZE,
            )
            v_rot = _load_tq4_rotated_chunk(
                value_cache,
                block_table,
                value_codebook,
                seq_idx,
                kv_head_idx,
                seq_offsets,
                dims,
                block_table_stride,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                HEAD_SIZE,
            )
            k_norm2 = tl.sum(k_rot * k_rot, axis=0)
            v_norm2 = tl.sum(v_rot * v_rot, axis=0)
            k_scale = _load_tq4_magnitude(
                key_cache,
                block_table,
                seq_idx,
                kv_head_idx,
                seq_offsets,
                block_table_stride,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                HEAD_SIZE,
            ) / tl.maximum(tl.sqrt(k_norm2), 1.0e-8)
            v_scale = _load_tq4_magnitude(
                value_cache,
                block_table,
                seq_idx,
                kv_head_idx,
                seq_offsets,
                block_table_stride,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                HEAD_SIZE,
            ) / tl.maximum(tl.sqrt(v_norm2), 1.0e-8)

            scores = tl.sum(q_rot[:, None] * k_rot, axis=0)
            scores = scores * k_scale * softmax_scale
            if LOGIT_CAP > 0:
                scores = LOGIT_CAP * _tanh(scores / LOGIT_CAP)
            scores = tl.where(tile_mask, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            acc_rot = acc_rot * alpha + tl.sum(v_rot * (p * v_scale)[None, :], axis=1)
            m_i = m_new
            l_i = l_new

        acc_rot = acc_rot / l_i
        h_out = _hadamard_sign(dims[:, None], dims[None, :])
        out = tl.sum(h_out * acc_rot[None, :], axis=1) * INV_SQRT_HEAD
        out = out * tl.load(value_signs + dims).to(tl.float32)
        tl.store(
            output
            + query_idx * output_stride_0
            + query_head_idx * output_stride_1
            + dims,
            out,
        )

    @triton.jit
    def _load_tq4_rotated_chunk(
        cache,
        block_table,
        codebook,
        seq_idx,
        kv_head_idx,
        seq_offsets,
        dims,
        block_table_stride: tl.constexpr,
        stride_cache_0: tl.constexpr,
        stride_cache_1: tl.constexpr,
        stride_cache_2: tl.constexpr,
        stride_cache_3: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
    ):
        physical_block_idx = tl.load(
            block_table + seq_idx * block_table_stride + seq_offsets // BLOCK_SIZE
        ).to(tl.int64)
        block_offsets = seq_offsets % BLOCK_SIZE
        byte_offsets = dims // 2
        packed = tl.load(
            cache
            + physical_block_idx[None, :] * stride_cache_0
            + block_offsets[None, :] * stride_cache_1
            + kv_head_idx * stride_cache_2
            + byte_offsets[:, None] * stride_cache_3,
            mask=dims[:, None] < HEAD_SIZE,
            other=0,
        ).to(tl.uint8)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        qidx = tl.where((dims[:, None] & 1) == 0, low, high)
        return tl.load(codebook + qidx.to(tl.int64)).to(tl.float32)

    @triton.jit
    def _load_tq4_magnitude(
        cache,
        block_table,
        seq_idx,
        kv_head_idx,
        seq_offsets,
        block_table_stride: tl.constexpr,
        stride_cache_0: tl.constexpr,
        stride_cache_1: tl.constexpr,
        stride_cache_2: tl.constexpr,
        stride_cache_3: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
    ):
        physical_block_idx = tl.load(
            block_table + seq_idx * block_table_stride + seq_offsets // BLOCK_SIZE
        ).to(tl.int64)
        block_offsets = seq_offsets % BLOCK_SIZE
        packed_width = (HEAD_SIZE + 1) // 2
        mag_lo = tl.load(
            cache
            + physical_block_idx * stride_cache_0
            + block_offsets * stride_cache_1
            + kv_head_idx * stride_cache_2
            + (packed_width + 0) * stride_cache_3
        ).to(tl.uint16)
        mag_hi = tl.load(
            cache
            + physical_block_idx * stride_cache_0
            + block_offsets * stride_cache_1
            + kv_head_idx * stride_cache_2
            + (packed_width + 1) * stride_cache_3
        ).to(tl.uint16)
        return (mag_lo | (mag_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    @triton.jit
    def _tq4_decode_attention_chunked_kernel(
        output,
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        key_codebook,
        key_signs,
        value_codebook,
        value_signs,
        softmax_scale: tl.constexpr,
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        block_table_stride: tl.constexpr,
        query_stride_0: tl.constexpr,
        query_stride_1: tl.constexpr,
        output_stride_0: tl.constexpr,
        output_stride_1: tl.constexpr,
        stride_k_cache_0: tl.constexpr,
        stride_k_cache_1: tl.constexpr,
        stride_k_cache_2: tl.constexpr,
        stride_k_cache_3: tl.constexpr,
        stride_v_cache_0: tl.constexpr,
        stride_v_cache_1: tl.constexpr,
        stride_v_cache_2: tl.constexpr,
        stride_v_cache_3: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        D_CHUNK: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        INV_SQRT_HEAD: tl.constexpr,
        LOGIT_CAP: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        query_head_idx = tl.program_id(1)
        kv_head_idx = query_head_idx // num_queries_per_kv

        full_dims = tl.arange(0, HEAD_SIZE)
        chunk_dims = tl.arange(0, D_CHUNK)
        tile = tl.arange(0, TILE_SIZE)

        query_idx = tl.load(query_start_loc + seq_idx)
        q_full = tl.load(
            query
            + query_idx * query_stride_0
            + query_head_idx * query_stride_1
            + full_dims,
        ).to(tl.float32)
        q_signed = q_full * tl.load(key_signs + full_dims).to(tl.float32)

        seq_len = tl.load(seq_lens + seq_idx)
        start_pos = 0
        if SLIDING_WINDOW > 0:
            start_pos = tl.maximum(0, seq_len - SLIDING_WINDOW)

        m_i = tl.full((), float("-inf"), dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)
        acc_rot = tl.zeros((HEAD_SIZE,), dtype=tl.float32)

        for tile_start in range(start_pos, seq_len, TILE_SIZE):
            seq_offsets = tile_start + tile
            tile_mask = seq_offsets < seq_len
            k_rot_full = _load_tq4_rotated_chunk(
                key_cache, block_table, key_codebook, seq_idx, kv_head_idx,
                seq_offsets, full_dims, block_table_stride, stride_k_cache_0,
                stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
                BLOCK_SIZE, HEAD_SIZE,
            )
            v_rot_full = _load_tq4_rotated_chunk(
                value_cache, block_table, value_codebook, seq_idx, kv_head_idx,
                seq_offsets, full_dims, block_table_stride, stride_v_cache_0,
                stride_v_cache_1, stride_v_cache_2, stride_v_cache_3,
                BLOCK_SIZE, HEAD_SIZE,
            )
            k_norm2 = tl.sum(k_rot_full * k_rot_full, axis=0)
            v_norm2 = tl.sum(v_rot_full * v_rot_full, axis=0)

            k_scale = _load_tq4_magnitude(
                key_cache, block_table, seq_idx, kv_head_idx, seq_offsets,
                block_table_stride, stride_k_cache_0, stride_k_cache_1,
                stride_k_cache_2, stride_k_cache_3, BLOCK_SIZE, HEAD_SIZE,
            ) / tl.maximum(tl.sqrt(k_norm2), 1.0e-8)
            v_scale = _load_tq4_magnitude(
                value_cache, block_table, seq_idx, kv_head_idx, seq_offsets,
                block_table_stride, stride_v_cache_0, stride_v_cache_1,
                stride_v_cache_2, stride_v_cache_3, BLOCK_SIZE, HEAD_SIZE,
            ) / tl.maximum(tl.sqrt(v_norm2), 1.0e-8)

            scores = tl.zeros((TILE_SIZE,), dtype=tl.float32)
            for d_start in range(0, HEAD_SIZE, D_CHUNK):
                dims = d_start + chunk_dims
                h = _hadamard_sign(dims[:, None], full_dims[None, :])
                q_rot = tl.sum(h * q_signed[None, :], axis=1) * INV_SQRT_HEAD
                k_rot = _load_tq4_rotated_chunk(
                    key_cache, block_table, key_codebook, seq_idx, kv_head_idx,
                    seq_offsets, dims, block_table_stride, stride_k_cache_0,
                    stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
                    BLOCK_SIZE, HEAD_SIZE,
                )
                scores += tl.sum(q_rot[:, None] * k_rot, axis=0)

            scores = scores * k_scale * softmax_scale
            if LOGIT_CAP > 0:
                scores = LOGIT_CAP * _tanh(scores / LOGIT_CAP)
            scores = tl.where(tile_mask, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            acc_rot = acc_rot * alpha

            acc_rot += tl.sum(v_rot_full * (p * v_scale)[None, :], axis=1)

            m_i = m_new
            l_i = l_new

        acc_rot = acc_rot / l_i

        for d_start in range(0, HEAD_SIZE, D_CHUNK):
            dims = d_start + chunk_dims
            h = _hadamard_sign(dims[:, None], full_dims[None, :])
            out_chunk = tl.sum(h * acc_rot[None, :], axis=1) * INV_SQRT_HEAD
            out_chunk = out_chunk * tl.load(value_signs + dims).to(tl.float32)
            tl.store(
                output
                + query_idx * output_stride_0
                + query_head_idx * output_stride_1
                + dims,
                out_chunk,
            )


def encode_tq4(x: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Encode [tokens, kv_heads, head_size] fp tensor to packed uint8."""
    if x.numel() == 0:
        return torch.empty(*x.shape[:-1], packed_tq4_width(x.shape[-1]), dtype=torch.uint8, device=x.device)
    head_size = x.shape[-1]
    _, boundaries, signs, padded_dim = _params(head_size, x.device, seed)
    flat = x.reshape(-1, head_size).float()
    magnitudes, indices, _ = _polar_encode(flat, signs, boundaries, padded_dim)
    packed = _pack_nibbles(indices)
    mag_bytes = _magnitude_to_bytes(magnitudes)
    encoded = torch.cat([packed, mag_bytes], dim=-1)
    return encoded.reshape(*x.shape[:-1], encoded.shape[-1])


def decode_tq4(encoded: torch.Tensor, *, head_size: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Decode packed uint8 tensor back to [*, head_size] fp tensor."""
    if encoded.numel() == 0:
        return torch.empty(*encoded.shape[:-1], head_size, dtype=dtype, device=encoded.device)
    codebook, _, signs, padded_dim = _params(head_size, encoded.device, seed)
    packed_width = (head_size + 1) // 2
    flat = encoded.reshape(-1, encoded.shape[-1])
    packed = flat[:, :packed_width]
    mag_bytes = flat[:, packed_width : packed_width + META_BYTES]
    indices = _unpack_nibbles(packed, padded_dim)
    magnitudes = _bytes_to_magnitude(mag_bytes)
    decoded = _polar_decode(magnitudes, indices, codebook, signs, head_size, padded_dim)
    return decoded.reshape(*encoded.shape[:-1], head_size).to(dtype)


def tq4_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    _debug_log_path(
        "cache_update "
        f"key={tuple(key.shape)} value={tuple(value.shape)} "
        f"kv_cache={tuple(kv_cache.shape)} slots={tuple(slot_mapping.shape)}"
    )
    key_cache, value_cache = kv_cache.unbind(1)
    key_flat = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])
    value_flat = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])
    encoded_key = encode_tq4(key, seed=42)
    encoded_value = encode_tq4(value, seed=43)
    key_flat.index_copy_(0, slot_mapping, encoded_key)
    value_flat.index_copy_(0, slot_mapping, encoded_value)
    _mark_tq4_blocks_dirty(kv_cache, slot_mapping)

    if _debug_compare_enabled() and slot_mapping.numel() > 0:
        sample_n = min(4, int(slot_mapping.numel()))
        sample_local = slice(0, sample_n)
        sample_slots = slot_mapping[sample_local]
        decoded_key = decode_tq4(
            key_flat.index_select(0, sample_slots),
            head_size=key.shape[-1],
            seed=42,
            dtype=key.dtype,
        )
        decoded_value = decode_tq4(
            value_flat.index_select(0, sample_slots),
            head_size=value.shape[-1],
            seed=43,
            dtype=value.dtype,
        )
        _summarize_compare("cache_update_key", key[sample_local], decoded_key)
        _summarize_compare("cache_update_value", value[sample_local], decoded_value)


def decode_tq4_kv_cache(kv_cache: torch.Tensor, *, head_size: int, dtype: torch.dtype) -> torch.Tensor:
    key_cache, value_cache = kv_cache.unbind(1)
    key = decode_tq4(key_cache, head_size=head_size, seed=42, dtype=dtype)
    value = decode_tq4(value_cache, head_size=head_size, seed=43, dtype=dtype)
    return torch.stack((key, value), dim=1).contiguous()


def decode_tq4_referenced_kv_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    head_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    used = torch.unique(block_table[block_table >= 0]).to(torch.long)
    if used.numel() == 0:
        compact_shape = (
            0,
            kv_cache.shape[1],
            kv_cache.shape[2],
            kv_cache.shape[3],
            head_size,
        )
        return (
            torch.empty(compact_shape, dtype=dtype, device=kv_cache.device),
            block_table.clone(),
        )
    _debug_log_path(
        "decode_referenced_cache "
        f"kv_cache={tuple(kv_cache.shape)} used_blocks={int(used.numel())} "
        f"block_table={tuple(block_table.shape)} head_size={head_size}"
    )
    state = _get_decode_cache_state(kv_cache, head_size=head_size, dtype=dtype)
    block_versions = state["block_versions"]
    decoded_versions = state["decoded_versions"]
    dirty_mask = decoded_versions.index_select(0, used) != block_versions.index_select(0, used)
    dirty_blocks = used[dirty_mask]

    if dirty_blocks.numel() > 0:
        compact_cache = kv_cache.index_select(0, dirty_blocks)
        state["decoded_cache"].index_copy_(
            0,
            dirty_blocks,
            decode_tq4_kv_cache(compact_cache, head_size=head_size, dtype=dtype),
        )
        decoded_versions.index_copy_(0, dirty_blocks, block_versions.index_select(0, dirty_blocks))

    decoded = state["decoded_cache"].index_select(0, used)

    remap = torch.empty(int(used.max().item()) + 1, dtype=torch.long, device=block_table.device)
    remap[used] = torch.arange(used.numel(), dtype=torch.long, device=block_table.device)
    compact_table = block_table.clone()
    valid = compact_table >= 0
    compact_table[valid] = remap[compact_table[valid].to(torch.long)].to(compact_table.dtype)
    return decoded, compact_table


def _can_use_live_full_prefill(
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    attn_metadata: Any,
) -> bool:
    if key is None or value is None:
        return False
    if getattr(attn_metadata, "use_cascade", False):
        return False
    if getattr(attn_metadata, "max_query_len", 1) <= 1:
        return False
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None:
        return False
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", key.shape[0]))
    if key.shape[0] < num_actual_tokens or value.shape[0] < num_actual_tokens:
        return False
    return int(seq_lens.sum().item()) == num_actual_tokens


def _context_attention_fwd_compat(
    prefill_attn: Any,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool,
    softmax_scale: float,
    sliding_window_q: int | None,
    sliding_window_k: int | None,
    block_size: int = 32,
) -> None:
    """Run vLLM's prefill kernel with a smaller block for T4 compatibility."""
    lq, lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
    scale = softmax_scale * prefill_attn.RCP_LN2
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    grid = (batch, head, prefill_attn.triton.cdiv(max_input_len, block_size))
    prefill_attn._fwd_kernel[grid](
        q,
        k,
        v,
        scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=block_size,
        BLOCK_DMODEL=prefill_attn.triton.next_power_of_2(lk),
        BLOCK_N=block_size,
        IS_CAUSAL=is_causal,
        SLIDING_WINDOW_Q=sliding_window_q if sliding_window_q is not None else 0,
        SLIDING_WINDOW_K=sliding_window_k if sliding_window_k is not None else 0,
        num_warps=4,
        num_stages=1,
        Lk=lk,
    )


def _try_fused_tq4_decode_attention(
    self: Any,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    output_block_scale: torch.Tensor | None,
) -> bool:
    def reject(reason: str) -> bool:
        if os.environ.get(_FUSED_DEBUG_ENV) == "1":
            print(f"TurboQuant fused decode disabled: {reason}", flush=True)
        return False

    if triton is None or output is None:
        return reject("triton/output unavailable")
    if os.environ.get(_FUSED_DECODE_ENV, "1") == "0":
        return reject("env disabled")
    if output_scale is not None or output_block_scale is not None:
        return reject("output quantization requested")
    if not _supports_fused_tq4_decode(self.head_size, getattr(self, "head_size_v", self.head_size)):
        return reject(f"unsupported head size {self.head_size}")
    if getattr(attn_metadata, "max_query_len", 0) != 1:
        return reject(f"max_query_len={getattr(attn_metadata, 'max_query_len', None)}")
    if getattr(attn_metadata, "use_cascade", False):
        return reject("cascade attention")
    if getattr(self, "alibi_slopes", None) is not None:
        return reject("alibi")
    if getattr(self, "sinks", None) is not None:
        return reject("sinks")
    if getattr(attn_metadata, "mm_prefix_range", None) is not None:
        return reject("mm prefix")

    try:
        key_cache, value_cache = kv_cache.unbind(1)
        key_codebook, _, key_signs, _ = _params(self.head_size, query.device, 42)
        value_codebook, _, value_signs, _ = _params(self.head_size, query.device, 43)
        sliding_window = self.sliding_window[0] + 1 if self.sliding_window[0] >= 0 else 0
        kernel = _tq4_decode_attention_chunked_kernel if self.head_size == 256 else _tq4_decode_attention_kernel
        kwargs = dict(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            key_codebook=key_codebook,
            key_signs=key_signs,
            value_codebook=value_codebook,
            value_signs=value_signs,
            softmax_scale=self.scale,
            num_query_heads=self.num_heads,
            num_queries_per_kv=self.num_queries_per_kv,
            block_table_stride=attn_metadata.block_table.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            BLOCK_SIZE=key_cache.shape[1],
            HEAD_SIZE=self.head_size,
            TILE_SIZE=8,
            SLIDING_WINDOW=sliding_window,
            INV_SQRT_HEAD=self.head_size ** -0.5,
            LOGIT_CAP=float(getattr(self, "logits_soft_cap", 0.0) or 0.0),
        )
        if self.head_size == 256:
            kwargs["D_CHUNK"] = 32
            kwargs["TILE_SIZE"] = 4
        kernel[(attn_metadata.seq_lens.shape[0], self.num_heads)](
            **kwargs,
            num_warps=8,
            num_stages=1,
        )
        if os.environ.get(_FUSED_DEBUG_ENV) == "1":
            print("TurboQuant fused decode used", flush=True)
        return True
    except Exception as exc:
        if os.environ.get(_FUSED_DEBUG_ENV) == "1":
            print(f"TurboQuant fused decode failed: {type(exc).__name__}: {exc}", flush=True)
        os.environ[_FUSED_DECODE_ENV] = "0"
        return False


def patch_vllm_tq4_runtime() -> None:
    """Install monkeypatches for vLLM's Triton attention backend."""
    try:
        import vllm.v1.attention.backends.triton_attn as triton_attn
        import vllm.v1.attention.ops.triton_prefill_attention as prefill_attn
        import vllm.v1.core.kv_cache_utils as kv_cache_utils
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    except Exception:
        return

    backend = triton_attn.TritonAttentionBackend
    for dtype in ("turboquant", "turboquant_4bit"):
        if dtype not in backend.supported_kv_cache_dtypes:
            backend.supported_kv_cache_dtypes.append(dtype)

    impl_cls = triton_attn.TritonAttentionImpl
    if not getattr(impl_cls, "_turboquant_tq4_patched", False):
        orig_update = impl_cls.do_kv_cache_update
        orig_forward = impl_cls.forward

        def patched_update(self: Any, layer: Any, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor):
            if is_tq4_dtype(getattr(self, "kv_cache_dtype", "")):
                if self.attn_type in (triton_attn.AttentionType.ENCODER_ONLY, triton_attn.AttentionType.ENCODER):
                    return
                tq4_cache_update(key, value, kv_cache, slot_mapping)
                return
            return orig_update(self, layer, key, value, kv_cache, slot_mapping)

        def patched_forward(self: Any, layer: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: Any, output: torch.Tensor | None = None, output_scale: torch.Tensor | None = None, output_block_scale: torch.Tensor | None = None):
            if is_tq4_dtype(getattr(self, "kv_cache_dtype", "")) and attn_metadata is not None:
                _debug_log_path(
                    "forward_enter "
                    f"query={tuple(query.shape)} key={tuple(key.shape) if key is not None else None} "
                    f"value={tuple(value.shape) if value is not None else None} "
                    f"max_query_len={getattr(attn_metadata, 'max_query_len', None)} "
                    f"num_actual_tokens={getattr(attn_metadata, 'num_actual_tokens', None)}"
                )
                if (
                    output is not None
                    and output_scale is None
                    and output_block_scale is None
                    and _can_use_live_full_prefill(key, value, attn_metadata)
                ):
                    num_actual_tokens = attn_metadata.num_actual_tokens
                    _debug_log_path(
                        "forward_branch=live_full_prefill "
                        f"tokens={num_actual_tokens} seq_lens={tuple(attn_metadata.seq_lens.shape)}"
                    )
                    _context_attention_fwd_compat(
                        prefill_attn,
                        q=query[:num_actual_tokens],
                        k=key[:num_actual_tokens],
                        v=value[:num_actual_tokens],
                        o=output[:num_actual_tokens],
                        b_start_loc=attn_metadata.query_start_loc,
                        b_seq_len=attn_metadata.seq_lens,
                        max_input_len=attn_metadata.max_query_len,
                        is_causal=True,
                        softmax_scale=self.scale,
                        sliding_window_q=self.sliding_window[0],
                        sliding_window_k=self.sliding_window[1],
                    )
                    return output
                else:
                    _debug_log_path("forward_branch=decode_path")
                    if _try_fused_tq4_decode_attention(
                        self,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        output_scale,
                        output_block_scale,
                    ):
                        _debug_log_path("forward_decode_subpath=fused_decode")
                        return output
                    _debug_log_path("forward_decode_subpath=fp16_fallback")
                    kv_cache, compact_block_table = decode_tq4_referenced_kv_cache(
                        kv_cache,
                        attn_metadata.block_table,
                        head_size=self.head_size,
                        dtype=query.dtype,
                    )
                    attn_metadata = replace(attn_metadata, block_table=compact_block_table)
            return orig_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        impl_cls.do_kv_cache_update = patched_update
        impl_cls.forward = patched_forward
        impl_cls._turboquant_tq4_patched = True

    if not getattr(Attention, "_turboquant_tq4_spec_patched", False):
        orig_get_spec = Attention.get_kv_cache_spec

        def patched_get_kv_cache_spec(self: Any, vllm_config: Any):
            if is_tq4_dtype(getattr(self, "kv_cache_dtype", "")):
                block_size = tq4_block_size(
                    vllm_config.cache_config.block_size,
                    self.num_kv_heads,
                    self.head_size,
                    self.head_size_v,
                )
                packed_width = packed_tq4_width(self.head_size)
                if self.sliding_window is not None:
                    return SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=self.num_kv_heads,
                        head_size=packed_width,
                        dtype=torch.uint8,
                        sliding_window=self.sliding_window,
                    )
                return FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_size=packed_width,
                    head_size_v=packed_width,
                    dtype=torch.uint8,
                )
            return orig_get_spec(self, vllm_config)

        Attention.get_kv_cache_spec = patched_get_kv_cache_spec
        Attention._turboquant_tq4_spec_patched = True

    if not getattr(kv_cache_utils, "_turboquant_page_unify_patched", False):
        orig_unify = kv_cache_utils.unify_kv_cache_spec_page_size
        orig_get_uniform_page_size = kv_cache_utils.get_uniform_page_size

        def patched_unify_kv_cache_spec_page_size(kv_cache_spec: dict[str, Any]) -> dict[str, Any]:
            try:
                return orig_unify(kv_cache_spec)
            except NotImplementedError:
                # vLLM's default fallback only adjusts block_size. Hybrid
                # Qwen/Mamba models can still produce non-divisible page sizes
                # after packed TQ4 changes attention bytes/token. In that case
                # scale block sizes to the least common page size. Avoid using
                # page_size_padded here: vLLM's tensor reshape path asks the
                # attention backend for a shape based on real block dimensions.
                common_page_size = lcm(*(layer.page_size_bytes for layer in kv_cache_spec.values()))
                unified: dict[str, Any] = {}
                for layer_name, layer_spec in kv_cache_spec.items():
                    ratio = common_page_size // layer_spec.page_size_bytes
                    unified[layer_name] = replace(layer_spec, block_size=layer_spec.block_size * ratio)
                return unified

        def patched_get_uniform_page_size(kv_cache_specs: Any) -> int:
            try:
                return orig_get_uniform_page_size(kv_cache_specs)
            except AssertionError:
                page_sizes = [layer.page_size_bytes for layer in kv_cache_specs]
                return max(page_sizes)

        kv_cache_utils.unify_kv_cache_spec_page_size = patched_unify_kv_cache_spec_page_size
        kv_cache_utils.get_uniform_page_size = patched_get_uniform_page_size
        kv_cache_utils._turboquant_page_unify_patched = True
