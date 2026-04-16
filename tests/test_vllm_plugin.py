"""Tests for vLLM plugin integration."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import turboquant.vllm_plugin as vllm_plugin
from turboquant.vllm_plugin import TurboQuantKVHook


FUSED_DECODE_MODEL_PROFILES = [
    {
        "label": "qwen3.5-4b",
        "seq_len": 24,
        "block_size": 16,
        "num_blocks": 2,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_size": 256,
        "sliding_window": (-1, 0),
        "logits_soft_cap": 0.0,
    },
    {
        "label": "qwen3.5-9b",
        "seq_len": 1024,
        "block_size": 16,
        "num_blocks": 64,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_size": 256,
        "sliding_window": (-1, 0),
        "logits_soft_cap": 0.0,
    },
    {
        "label": "gemma-4-e4b-it",
        "seq_len": 768,
        "block_size": 16,
        "num_blocks": 48,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_size": 256,
        "sliding_window": (511, 0),
        "logits_soft_cap": 0.0,
    },
    {
        "label": "bonsai-8b-1bit",
        "seq_len": 1024,
        "block_size": 16,
        "num_blocks": 64,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "sliding_window": (-1, 0),
        "logits_soft_cap": 0.0,
    },
]


class TestTurboQuantKVHook:
    """Test the standalone KV hook (works without vLLM installed)."""

    @pytest.fixture
    def hook(self):
        return TurboQuantKVHook(
            num_heads=32, num_kv_heads=8, head_dim=128,
            kv_bits=4, device="cpu",
        )

    def test_encode_decode_keys_shape(self, hook):
        """vLLM format: (batch, heads, seq, dim)."""
        keys = torch.randn(1, 8, 64, 128)
        encoded = hook.encode_keys(keys, layer_idx=0)
        decoded = hook.decode_keys(encoded, layer_idx=0)
        assert decoded.shape == keys.shape

    def test_encode_decode_values_shape(self, hook):
        values = torch.randn(1, 8, 64, 128)
        encoded = hook.encode_values(values, layer_idx=0)
        decoded = hook.decode_values(encoded, layer_idx=0)
        assert decoded.shape == values.shape

    def test_quality_preserved(self, hook):
        torch.manual_seed(42)
        keys = torch.randn(1, 8, 256, 128)
        encoded = hook.encode_keys(keys, layer_idx=0)
        decoded = hook.decode_keys(encoded, layer_idx=0)

        cos_sim = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, 128), decoded.reshape(-1, 128), dim=-1
        )
        assert cos_sim.mean() > 0.99, f"Quality too low: {cos_sim.mean():.4f}"

    def test_multi_layer(self, hook):
        """Different layers should be independent."""
        k0 = torch.randn(1, 8, 32, 128)
        k1 = torch.randn(1, 8, 32, 128)

        hook.encode_keys(k0, layer_idx=0)
        hook.encode_keys(k1, layer_idx=1)

        d0 = hook.decode_keys(hook.encode_keys(k0, layer_idx=0), layer_idx=0)
        d1 = hook.decode_keys(hook.encode_keys(k1, layer_idx=1), layer_idx=1)

        assert d0.shape == k0.shape
        assert d1.shape == k1.shape

    def test_no_nan_inf(self, hook):
        keys = torch.randn(1, 8, 128, 128)
        encoded = hook.encode_keys(keys)
        decoded = hook.decode_keys(encoded)
        assert not torch.isnan(decoded).any()
        assert not torch.isinf(decoded).any()

    def test_vllm_format_batch(self, hook):
        """Batch size > 1."""
        keys = torch.randn(4, 8, 64, 128)
        encoded = hook.encode_keys(keys)
        decoded = hook.decode_keys(encoded)
        assert decoded.shape == (4, 8, 64, 128)


class TestPluginRegistration:
    """Test that registration function is safe to call."""

    def test_register_idempotent(self):
        vllm_plugin.register()
        vllm_plugin.register()  # Should not error on second call

    def test_register_without_vllm(self):
        """Should not crash when vLLM is not installed."""
        vllm_plugin.register()  # Graceful fallback

    def test_register_applies_runtime_patch(self, monkeypatch):
        calls = []

        monkeypatch.setattr(vllm_plugin, "_registered", False)
        monkeypatch.setattr(
            vllm_plugin, "_register_quantization", lambda: calls.append("quant")
        )
        monkeypatch.setattr(
            vllm_plugin, "_register_cache_dtype", lambda: calls.append("dtype")
        )
        monkeypatch.setattr(
            vllm_plugin,
            "_ensure_runtime_patch",
            lambda: calls.append("runtime"),
        )

        vllm_plugin.register()

        assert calls == ["quant", "dtype", "runtime"]

    def test_turboquant_dtype_resolution_triggers_runtime_patch(self, monkeypatch):
        torch_utils = importlib.import_module("vllm.utils.torch_utils")
        arg_utils = importlib.import_module("vllm.engine.arg_utils")

        original_resolve = torch_utils.resolve_kv_cache_dtype_string
        original_arg_resolve = arg_utils.resolve_kv_cache_dtype_string
        original_flag = getattr(
            torch_utils, "_turboquant_dtype_resolution_patched", False
        )

        calls = []

        monkeypatch.setattr(vllm_plugin, "_runtime_patch_attempted", False)
        monkeypatch.setattr(
            vllm_plugin,
            "_ensure_runtime_patch",
            lambda: calls.append("runtime"),
        )
        monkeypatch.setattr(
            torch_utils,
            "_turboquant_dtype_resolution_patched",
            False,
            raising=False,
        )

        try:
            vllm_plugin._register_cache_dtype()
            resolved = torch_utils.resolve_kv_cache_dtype_string(
                "turboquant_4bit", None
            )
            assert resolved == "turboquant_4bit"
            assert calls == ["runtime"]
        finally:
            monkeypatch.setattr(
                torch_utils,
                "resolve_kv_cache_dtype_string",
                original_resolve,
            )
            monkeypatch.setattr(
                arg_utils,
                "resolve_kv_cache_dtype_string",
                original_arg_resolve,
            )
            monkeypatch.setattr(
                torch_utils,
                "_turboquant_dtype_resolution_patched",
                original_flag,
                raising=False,
            )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "profile",
    FUSED_DECODE_MODEL_PROFILES,
    ids=[profile["label"] for profile in FUSED_DECODE_MODEL_PROFILES],
)
def test_fused_decode_matches_reference_attention(profile):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    runtime = importlib.import_module("turboquant.vllm_tq4_runtime")
    if runtime.triton is None:
        pytest.skip("Triton runtime unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_len = profile["seq_len"]
    block_size = profile["block_size"]
    num_blocks = profile["num_blocks"]
    num_heads = profile["num_heads"]
    num_kv_heads = profile["num_kv_heads"]
    head_size = profile["head_size"]
    packed_width = runtime.packed_tq4_width(head_size)
    dtype = torch.float16

    query = torch.randn(1, num_heads, head_size, device=device, dtype=dtype)
    key = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)

    kv_cache = torch.zeros(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        packed_width,
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.long)
    runtime.tq4_cache_update(key, value, kv_cache, slot_mapping)

    attn_metadata = SimpleNamespace(
        block_table=torch.tensor(
            [list(range(num_blocks))], device=device, dtype=torch.int32
        ),
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0], device=device, dtype=torch.int32),
        max_query_len=1,
        num_actual_tokens=1,
        use_cascade=False,
        mm_prefix_range=None,
    )
    impl = SimpleNamespace(
        head_size=head_size,
        num_heads=num_heads,
        num_queries_per_kv=num_heads // num_kv_heads,
        sliding_window=profile["sliding_window"],
        alibi_slopes=None,
        sinks=None,
        logits_soft_cap=profile["logits_soft_cap"],
        scale=head_size ** -0.5,
    )

    fused_output = torch.empty_like(query)
    used_fused = runtime._try_fused_tq4_decode_attention(
        impl,
        query,
        kv_cache,
        attn_metadata,
        fused_output,
        None,
        None,
    )
    assert used_fused, "Expected fused decode path to run"

    decoded_cache, _ = runtime.decode_tq4_referenced_kv_cache(
        kv_cache,
        attn_metadata.block_table,
        head_size=head_size,
        dtype=dtype,
    )
    decoded_keys = decoded_cache[0, 0, :seq_len]
    decoded_values = decoded_cache[0, 1, :seq_len]

    ref = torch.empty_like(query)
    num_queries_per_kv = num_heads // num_kv_heads
    effective_window = impl.sliding_window[0] + 1 if impl.sliding_window[0] >= 0 else 0
    start = max(0, seq_len - effective_window) if effective_window > 0 else 0
    for q_head in range(num_heads):
        kv_head = q_head // num_queries_per_kv
        scores = torch.matmul(
            decoded_keys[start:, kv_head], query[0, q_head]
        ) * impl.scale
        if impl.logits_soft_cap:
            scores = impl.logits_soft_cap * torch.tanh(scores / impl.logits_soft_cap)
        probs = torch.softmax(scores.float(), dim=0).to(dtype)
        ref[0, q_head] = torch.sum(
            decoded_values[start:, kv_head] * probs[:, None], dim=0
        )

    max_abs_err = (fused_output - ref).abs().max().item()
    mean_abs_err = (fused_output - ref).abs().mean().item()
    assert max_abs_err < 0.25, f"max_abs_err={max_abs_err:.4f}, mean_abs_err={mean_abs_err:.4f}"


@pytest.mark.gpu
def test_fused_decode_matches_reference_attention_with_logit_soft_cap():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    runtime = importlib.import_module("turboquant.vllm_tq4_runtime")
    if runtime.triton is None:
        pytest.skip("Triton runtime unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_len = 768
    block_size = 16
    num_blocks = 48
    num_heads = 8
    num_kv_heads = 2
    head_size = 256
    logits_soft_cap = 50.0
    packed_width = runtime.packed_tq4_width(head_size)
    dtype = torch.float16

    query = torch.randn(1, num_heads, head_size, device=device, dtype=dtype)
    key = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)

    kv_cache = torch.zeros(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        packed_width,
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.long)
    runtime.tq4_cache_update(key, value, kv_cache, slot_mapping)

    attn_metadata = SimpleNamespace(
        block_table=torch.tensor(
            [list(range(num_blocks))], device=device, dtype=torch.int32
        ),
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0], device=device, dtype=torch.int32),
        max_query_len=1,
        num_actual_tokens=1,
        use_cascade=False,
        mm_prefix_range=None,
    )
    impl = SimpleNamespace(
        head_size=head_size,
        num_heads=num_heads,
        num_queries_per_kv=num_heads // num_kv_heads,
        sliding_window=(511, 0),
        alibi_slopes=None,
        sinks=None,
        logits_soft_cap=logits_soft_cap,
        scale=head_size ** -0.5,
    )

    fused_output = torch.empty_like(query)
    used_fused = runtime._try_fused_tq4_decode_attention(
        impl,
        query,
        kv_cache,
        attn_metadata,
        fused_output,
        None,
        None,
    )
    assert used_fused, "Expected fused decode path with logit soft cap to run"

    decoded_cache, _ = runtime.decode_tq4_referenced_kv_cache(
        kv_cache,
        attn_metadata.block_table,
        head_size=head_size,
        dtype=dtype,
    )
    decoded_keys = decoded_cache[0, 0]
    decoded_values = decoded_cache[0, 1]
    start = seq_len - 512

    ref = torch.empty_like(query)
    num_queries_per_kv = num_heads // num_kv_heads
    for q_head in range(num_heads):
        kv_head = q_head // num_queries_per_kv
        scores = torch.matmul(
            decoded_keys[start:, kv_head], query[0, q_head]
        ) * impl.scale
        scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
        probs = torch.softmax(scores.float(), dim=0).to(dtype)
        ref[0, q_head] = torch.sum(
            decoded_values[start:, kv_head] * probs[:, None], dim=0
        )

    max_abs_err = (fused_output - ref).abs().max().item()
    mean_abs_err = (fused_output - ref).abs().mean().item()
    assert max_abs_err < 0.25, f"max_abs_err={max_abs_err:.4f}, mean_abs_err={mean_abs_err:.4f}"


def test_decode_referenced_kv_cache_reuses_clean_blocks(monkeypatch):
    runtime = importlib.import_module("turboquant.vllm_tq4_runtime")
    runtime._DECODE_CACHE_STATE.clear()

    torch.manual_seed(0)
    block_size = 4
    num_blocks = 3
    num_kv_heads = 2
    head_size = 8
    packed_width = runtime.packed_tq4_width(head_size)
    dtype = torch.float16

    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, packed_width, dtype=torch.uint8
    )
    key = torch.randn(num_blocks * block_size, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn_like(key)
    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.long)
    runtime.tq4_cache_update(key, value, kv_cache, slot_mapping)

    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    decode_calls = []
    original_decode = runtime.decode_tq4_kv_cache

    def wrapped_decode(*args, **kwargs):
        decode_calls.append(args[0].shape[0])
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(runtime, "decode_tq4_kv_cache", wrapped_decode)

    decoded_first, compact_first = runtime.decode_tq4_referenced_kv_cache(
        kv_cache, block_table, head_size=head_size, dtype=dtype
    )
    decoded_second, compact_second = runtime.decode_tq4_referenced_kv_cache(
        kv_cache, block_table, head_size=head_size, dtype=dtype
    )

    assert decode_calls == [3]
    assert torch.equal(compact_first, compact_second)
    assert torch.allclose(decoded_first, decoded_second)


def test_decode_referenced_kv_cache_invalidates_only_written_blocks(monkeypatch):
    runtime = importlib.import_module("turboquant.vllm_tq4_runtime")
    runtime._DECODE_CACHE_STATE.clear()

    torch.manual_seed(0)
    block_size = 4
    num_blocks = 3
    num_kv_heads = 2
    head_size = 8
    packed_width = runtime.packed_tq4_width(head_size)
    dtype = torch.float16

    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, packed_width, dtype=torch.uint8
    )
    key = torch.randn(num_blocks * block_size, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn_like(key)
    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.long)
    runtime.tq4_cache_update(key, value, kv_cache, slot_mapping)

    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    runtime.decode_tq4_referenced_kv_cache(
        kv_cache, block_table, head_size=head_size, dtype=dtype
    )

    decode_calls = []
    original_decode = runtime.decode_tq4_kv_cache

    def wrapped_decode(*args, **kwargs):
        decode_calls.append(args[0].shape[0])
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(runtime, "decode_tq4_kv_cache", wrapped_decode)

    updated_key = torch.randn(block_size, num_kv_heads, head_size, dtype=dtype)
    updated_value = torch.randn_like(updated_key)
    updated_slots = torch.arange(block_size, 2 * block_size, dtype=torch.long)
    runtime.tq4_cache_update(updated_key, updated_value, kv_cache, updated_slots)
    decoded_after, _ = runtime.decode_tq4_referenced_kv_cache(
        kv_cache, block_table, head_size=head_size, dtype=dtype
    )

    assert decode_calls == [1]
    decoded_block = runtime.decode_tq4_kv_cache(
        kv_cache.index_select(0, torch.tensor([1])),
        head_size=head_size,
        dtype=dtype,
    )[0]
    assert torch.allclose(decoded_after[1], decoded_block)


def test_decode_referenced_kv_cache_handles_empty_block_table():
    runtime = importlib.import_module("turboquant.vllm_tq4_runtime")
    runtime._DECODE_CACHE_STATE.clear()

    block_size = 4
    num_blocks = 2
    num_kv_heads = 2
    head_size = 8
    packed_width = runtime.packed_tq4_width(head_size)
    dtype = torch.float16

    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, packed_width, dtype=torch.uint8
    )
    block_table = torch.full((1, 3), -1, dtype=torch.int32)

    decoded, compact = runtime.decode_tq4_referenced_kv_cache(
        kv_cache, block_table, head_size=head_size, dtype=dtype
    )

    assert decoded.shape == (0, 2, block_size, num_kv_heads, head_size)
    assert torch.equal(compact, block_table)
