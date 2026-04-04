"""TurboQuant: Efficient KV Cache Quantization for vLLM.

Implements Google's TurboQuant (PolarQuant + QJL) for 4-bit KV cache
compression, KIVI-style asymmetric quantization, and Bonsai-inspired
1-bit weight quantization for maximum inference efficiency.
"""

__version__ = "0.1.0"

from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer
from turboquant.core.kv_cache import QuantizedKVCache

__all__ = [
    "TurboQuantConfig",
    "TurboQuantizer",
    "QuantizedKVCache",
]
