"""vLLM plugin: register TurboQuant as a KV cache quantization method.

Enables:
    vllm serve model --quantization turboquant
    vllm serve model --kv-cache-dtype turboquant

This hooks into vLLM's attention layers to quantize KV cache entries
using PolarQuant + Hadamard rotation instead of FP16/FP8.

Installation:
    pip install turboquant-vllm

    # pyproject.toml entry point auto-registers on vLLM import:
    [project.entry-points."vllm.general_plugins"]
    turboquant = "turboquant.vllm_plugin:register"
"""

from __future__ import annotations

import logging
import torch
from typing import Any, Optional

logger = logging.getLogger(__name__)

_registered = False


def register():
    """Entry point called by vLLM's plugin system on startup."""
    global _registered
    if _registered:
        return
    _registered = True

    try:
        _register_quantization()
        _register_cache_dtype()
        logger.info("TurboQuant KV cache quantization registered with vLLM")
    except ImportError as e:
        logger.debug(f"vLLM not available, skipping plugin registration: {e}")
    except Exception as e:
        logger.warning(f"TurboQuant vLLM plugin registration failed: {e}")


def _register_quantization():
    """Register TurboQuantVLLMConfig with vLLM's quantization system."""
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    @register_quantization_config("turboquant")
    class TurboQuantVLLMConfig(QuantizationConfig):
        """vLLM quantization config for TurboQuant KV cache compression.

        Supports:
        - turbo_4bit: PolarQuant + Hadamard + Lloyd-Max (production)
        - turbo_3bit: PolarQuant + QJL residual correction
        - kivi: Asymmetric per-channel keys / per-token values
        """

        def __init__(
            self,
            kv_bits: int = 4,
            key_bits: Optional[int] = None,
            value_bits: Optional[int] = None,
            enable_hadamard: bool = True,
            enable_mixed_precision: bool = True,
            residual_buffer_size: int = 128,
        ):
            self.kv_bits = kv_bits
            self.key_bits = key_bits or kv_bits
            self.value_bits = value_bits or kv_bits
            self.enable_hadamard = enable_hadamard
            self.enable_mixed_precision = enable_mixed_precision
            self.residual_buffer_size = residual_buffer_size

        def get_name(self) -> str:
            return "turboquant"

        def get_supported_act_dtypes(self) -> list[torch.dtype]:
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 70  # Volta+ (T4 is sm_75)

        @staticmethod
        def get_config_filenames() -> list[str]:
            return ["turboquant_config.json"]

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> TurboQuantVLLMConfig:
            kv_bits = cls.get_from_keys_or(config, ["kv_bits", "bits"], 4)
            key_bits = cls.get_from_keys_or(config, ["key_bits"], None)
            value_bits = cls.get_from_keys_or(config, ["value_bits"], None)
            enable_hadamard = cls.get_from_keys_or(config, ["enable_hadamard"], True)
            return cls(
                kv_bits=kv_bits,
                key_bits=key_bits,
                value_bits=value_bits,
                enable_hadamard=enable_hadamard,
            )

        def get_quant_method(
            self, layer: torch.nn.Module, prefix: str
        ) -> Optional[QuantizeMethodBase]:
            from vllm.attention import Attention

            if isinstance(layer, Attention):
                return TurboQuantKVCacheMethod(self)
            return None

        def get_scaled_act_names(self) -> list[str]:
            return []

    class TurboQuantKVCacheMethod(BaseKVCacheMethod):
        """KV cache quantization method using TurboQuant algorithms.

        Hooks into vLLM's attention layer to:
        1. Intercept KV cache writes (encode: FP16 -> 4-bit)
        2. Intercept KV cache reads (decode: 4-bit -> FP16 for attention)
        """

        def __init__(self, quant_config: TurboQuantVLLMConfig):
            super().__init__(quant_config)
            self._quantizer = None
            self._fast_quantizer = None

        def create_weights(self, layer: torch.nn.Module) -> None:
            """Register TurboQuant parameters on the attention layer."""
            super().create_weights(layer)

            # Store config reference for later initialization
            layer.turboquant_config = self.quant_config

            # Pre-compute codebook as a buffer (not a parameter -- no gradients)
            from turboquant.core.lloyd_max import LloydMaxQuantizer

            head_dim = getattr(layer, "head_size", 128)
            padded_dim = 1 << (head_dim - 1).bit_length() if head_dim & (head_dim - 1) != 0 else head_dim

            lm = LloydMaxQuantizer(bits=self.quant_config.kv_bits, dim=padded_dim)
            layer.register_buffer(
                "turboquant_codebook", lm.codebook, persistent=False
            )
            layer.register_buffer(
                "turboquant_boundaries", lm.boundaries, persistent=False
            )

            # Hadamard random signs
            gen = torch.Generator(device="cpu").manual_seed(42)
            signs = (torch.randint(0, 2, (padded_dim,), generator=gen) * 2 - 1).float()
            layer.register_buffer(
                "turboquant_signs", signs, persistent=False
            )

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            """Initialize the fast quantizer after model is loaded."""
            super().process_weights_after_loading(layer)

            # Initialize the fast quantizer with model-specific dimensions
            from turboquant.config import TurboQuantConfig
            from turboquant.core.fast_quantizer import FastTurboQuantizer

            cfg = layer.turboquant_config
            head_dim = getattr(layer, "head_size", 128)
            num_kv_heads = getattr(layer, "num_kv_heads", 8)
            num_heads = getattr(layer, "num_heads", 32)

            tq_config = TurboQuantConfig.turbo_4bit(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            tq_config.kv_bits = cfg.kv_bits
            tq_config.key_bits = cfg.key_bits
            tq_config.value_bits = cfg.value_bits
            tq_config.enable_hadamard = cfg.enable_hadamard
            tq_config.enable_mixed_precision = cfg.enable_mixed_precision

            device = next(layer.parameters()).device
            layer.turboquant_quantizer = FastTurboQuantizer(
                tq_config, device=str(device)
            )

            logger.info(
                f"TurboQuant initialized: {cfg.kv_bits}-bit KV, "
                f"hadamard={cfg.enable_hadamard}, device={device}"
            )

    # Store references for external access
    register.TurboQuantVLLMConfig = TurboQuantVLLMConfig
    register.TurboQuantKVCacheMethod = TurboQuantKVCacheMethod


def _register_cache_dtype():
    """Register 'turboquant' as a valid kv_cache_dtype in vLLM."""
    try:
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        # TurboQuant stores quantized data as int16 (codebook indices)
        STR_DTYPE_TO_TORCH_DTYPE["turboquant"] = torch.int16
        STR_DTYPE_TO_TORCH_DTYPE["turboquant_4bit"] = torch.int16
        STR_DTYPE_TO_TORCH_DTYPE["turboquant_3bit"] = torch.int16
    except (ImportError, AttributeError):
        pass

    # Patch is_quantized_kv_cache to recognize turboquant
    try:
        import vllm.utils.torch_utils as torch_utils

        _orig_is_quantized = torch_utils.is_quantized_kv_cache

        def _patched_is_quantized(kv_cache_dtype: str) -> bool:
            if kv_cache_dtype.startswith("turboquant"):
                return True
            return _orig_is_quantized(kv_cache_dtype)

        torch_utils.is_quantized_kv_cache = _patched_is_quantized
    except (ImportError, AttributeError):
        pass


# ──────────────────────────────────────────────────────────────────────
#  Standalone encode/decode hooks for manual vLLM integration
# ──────────────────────────────────────────────────────────────────────

class TurboQuantKVHook:
    """Standalone KV cache encode/decode hook for manual integration.

    Use this when you can't use the plugin system (e.g., custom vLLM fork):

        hook = TurboQuantKVHook(num_heads=32, num_kv_heads=8, head_dim=128)

        # In attention forward:
        key_states = hook.encode_keys(key_states)
        value_states = hook.encode_values(value_states)
        # ... store in cache ...

        # Before attention computation:
        key_states = hook.decode_keys(cached_keys)
        value_states = hook.decode_values(cached_values)
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        kv_bits: int = 4,
        device: str = "cuda",
    ):
        from turboquant.config import TurboQuantConfig
        from turboquant.core.fast_quantizer import FastTurboQuantizer

        config = TurboQuantConfig.turbo_4bit(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        config.kv_bits = kv_bits
        self.quantizer = FastTurboQuantizer(config, device=device)
        self._meta_cache: dict[int, dict] = {}  # layer_idx -> meta

    def encode_keys(self, keys: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Quantize key states before cache storage.

        Args:
            keys: (batch, num_kv_heads, seq_len, head_dim) -- vLLM format
        Returns:
            Quantized keys (same shape, int16 dtype)
        """
        # vLLM uses (batch, heads, seq, dim), our API uses (batch, seq, heads, dim)
        k = keys.transpose(1, 2)
        q_k, meta = self.quantizer.encode_keys(k)
        self._meta_cache[("k", layer_idx)] = meta
        return q_k.transpose(1, 2)

    def encode_values(self, values: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        v = values.transpose(1, 2)
        q_v, meta = self.quantizer.encode_values(v)
        self._meta_cache[("v", layer_idx)] = meta
        return q_v.transpose(1, 2)

    def decode_keys(self, keys: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Dequantize key states before attention computation."""
        meta = self._meta_cache.get(("k", layer_idx), {})
        k = keys.transpose(1, 2)
        recon = self.quantizer.decode_keys(k, meta)
        return recon.transpose(1, 2)

    def decode_values(self, values: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        meta = self._meta_cache.get(("v", layer_idx), {})
        v = values.transpose(1, 2)
        recon = self.quantizer.decode_values(v, meta)
        return recon.transpose(1, 2)
