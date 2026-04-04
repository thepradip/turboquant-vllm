# TurboQuant-vLLM: Efficient KV Cache Quantization

**4-bit KV cache compression without quality loss** -- combining Google's TurboQuant, KIVI asymmetric quantization, and Bonsai-inspired 1-bit techniques for maximum inference efficiency.

## Key Features

| Feature | Description |
|---------|-------------|
| **TurboQuant (PolarQuant)** | Calibration-free online vector quantization: magnitude/direction decomposition + Hadamard rotation + Lloyd-Max codebook |
| **KIVI Asymmetric** | Per-channel key quantization (isolates outliers) + per-token value quantization (exploits sparse attention) |
| **Bonsai 1-bit (Q1_0_g128)** | Extreme 14x compression: 1 sign bit per weight + FP16 scale per 128 elements |
| **Mixed Precision** | Outlier channels at 8-bit, normal channels at 4-bit for fractional bit-widths |
| **QJL Residual Correction** | Optional Johnson-Lindenstrauss projection for residual error correction |
| **Hadamard Rotation** | Fast O(d log d) Walsh-Hadamard transform eliminates outliers pre-quantization |
| **Residual Buffer** | Recent tokens kept in FP16 for maximum quality on local context |
| **Quality Benchmarks** | Comprehensive evaluation suite with cosine similarity, attention error, needle-in-a-haystack, and dataset-based metrics |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TurboQuantizer                     │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  PolarQuant   │  │   KIVI   │  │  Bonsai 1-bit│  │
│  │  ┌──────────┐ │  │ per-chan  │  │  Q1_0_g128   │  │
│  │  │ Hadamard │ │  │ per-tok  │  │  sign + scale│  │
│  │  │ Rotation │ │  │          │  │              │  │
│  │  └────┬─────┘ │  └──────────┘  └──────────────┘  │
│  │  ┌────▼─────┐ │                                   │
│  │  │Lloyd-Max │ │  ┌──────────────────────────┐     │
│  │  │ Codebook │ │  │   Mixed Precision         │     │
│  │  └────┬─────┘ │  │   Outlier: 8-bit          │     │
│  │  ┌────▼─────┐ │  │   Normal:  4-bit          │     │
│  │  │   QJL    │ │  └──────────────────────────┘     │
│  │  │ Residual │ │                                   │
│  │  └──────────┘ │                                   │
│  └──────────────┘                                    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │         Quantized KV Cache Manager            │    │
│  │  ┌────────────────┐  ┌───────────────────┐   │    │
│  │  │ Quantized Cache │  │  Residual Buffer  │   │    │
│  │  │ (compressed)    │  │  (FP16, recent)   │   │    │
│  │  └────────────────┘  └───────────────────┘   │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## Quantization Presets

| Preset | Bits | Compression | Key Quality | Use Case |
|--------|------|-------------|-------------|----------|
| `turbo_4bit` | 4 | ~4x | >0.95 cosine sim | **Production** -- near-lossless |
| `turbo_3bit` | 3 | ~5.3x | >0.92 cosine sim | Long-context with good quality |
| `kivi_2bit` | 4K+2V | ~5.3x | >0.90 cosine sim | Maximum asymmetric savings |
| `onebit_extreme` | 1.125 | ~14x | >0.50 cosine sim | Edge/mobile, research |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run benchmarks
python -m benchmarks.quality_benchmark
python -m benchmarks.memory_benchmark
python -m benchmarks.dataset_eval
```

### Usage

```python
from turboquant import TurboQuantConfig, TurboQuantizer, QuantizedKVCache

# Choose a preset
config = TurboQuantConfig.turbo_4bit(
    num_heads=32, num_kv_heads=8, head_dim=128
)

# Create quantizer
quantizer = TurboQuantizer(config)

# Quantize KV cache entries
keys = torch.randn(1, 512, 8, 128)    # (batch, seq, kv_heads, dim)
values = torch.randn(1, 512, 8, 128)

# Encode (compress)
q_keys, key_meta = quantizer.encode_keys(keys)
q_vals, val_meta = quantizer.encode_values(values)

# Decode (decompress for attention)
recon_keys = quantizer.decode_keys(q_keys, key_meta)
recon_vals = quantizer.decode_values(q_vals, val_meta)

# Check quality
from turboquant.utils.metrics import QualityMetrics
report = QualityMetrics.full_report(keys, recon_keys, effective_bits=4.0)
print(f"Cosine similarity: {report.cosine_similarity_mean:.4f}")
print(f"SNR: {report.snr_db:.1f} dB")
```

### With KV Cache Manager

```python
# Full KV cache with residual buffer
cache = QuantizedKVCache(config, num_layers=32)
cache.set_quantizer(quantizer)

# Append tokens (auto-quantizes when buffer fills)
cache.append(layer_idx=0, keys=new_keys, values=new_values)

# Retrieve for attention (auto-dequantizes)
all_keys = cache.get_keys(layer_idx=0)
all_values = cache.get_values(layer_idx=0)

# Memory profiling
print(cache.memory_usage())
```

## Algorithm Details

### PolarQuant (TurboQuant Primary Stage)
1. **Decompose** each vector into magnitude `r` (FP16 scalar) and direction `d` (unit vector)
2. **Rotate** direction with randomized Hadamard transform -- redistributes outliers uniformly
3. **Quantize** rotated coordinates with fixed Lloyd-Max codebook (no per-block metadata needed)
4. **Pack** quantized indices into sub-byte storage

### KIVI Asymmetric Quantization
- **Keys: per-channel** -- outlier channels are persistent across tokens, so quantize along token dimension per channel
- **Values: per-token** -- no channel outlier pattern; attention is sparse, so quantize each token independently
- **Result**: 5x lower attention score error for keys, 15x lower for values vs. naive approaches

### Bonsai 1-bit (Q1_0_g128)
- Each weight = 1 sign bit: `0 → -scale`, `1 → +scale`
- Every 128 weights share one FP16 scale (mean absolute value)
- **Effective bits**: 1.125 per weight
- **Compression**: 14.2x vs FP16 (93% memory reduction)

## Papers & References

- **TurboQuant** (Google Research, ICLR 2026): PolarQuant + QJL for calibration-free KV cache compression
- **KIVI** (ICML 2024): Tuning-free asymmetric 2-bit KV quantization
- **KVQuant** (NeurIPS 2024): Non-uniform quantization for 10M context
- **QuaRot** (NeurIPS 2024): Outlier-free 4-bit inference via Hadamard rotations
- **QServe** (MLSys 2025): W4A8KV4 with SmoothAttention
- **Bonsai** (PrismML 2026): 1-bit Q1_0_g128 format for edge deployment

## License

Apache 2.0
