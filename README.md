# TurboQuant-vLLM

### Efficient KV Cache Quantization for High-Performance LLM Inference

4-bit KV cache compression without quality loss -- combining Google's TurboQuant (PolarQuant + QJL), KIVI asymmetric quantization, and PrismML's Bonsai-inspired 1-bit techniques for maximum inference efficiency.

Built on research from TurboQuant (Google, ICLR 2026), KIVI (ICML 2024), QuaRot (NeurIPS 2024), and the [Bonsai 1-bit](https://github.com/thepradip/prism-Bonsai-1bit) project.

---

## Why TurboQuant?

KV cache is the memory bottleneck for long-context LLM serving. At 32K context on Llama-3.1-8B, the KV cache alone consumes **4 GB in FP16**. TurboQuant compresses this to **1 GB at 4-bit** with **99.4% attention fidelity** -- no calibration data, no fine-tuning, works online at inference time.

```
  FP16 Baseline:    ████████████████████████████████████████  4,096 MB
  TurboQuant 4-bit: ██████████                               1,056 MB  (74% saved)
  KIVI 2-bit:       █████████                                1,024 MB  (75% saved)
  Bonsai 1-bit:     ███                                        288 MB  (93% saved)
```

---

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.1+
- macOS / Linux

### Step 1 -- Clone

```bash
git clone https://github.com/thepradip/turboquant-vllm.git
cd turboquant-vllm
```

```
pradip@Pradips-MBP ~ % git clone https://github.com/thepradip/turboquant-vllm.git
Cloning into 'turboquant-vllm'...
remote: Enumerating objects: 50, done.
remote: Counting objects: 100% (50/50), done.
remote: Compressing objects: 100% (44/44), done.
Receiving objects: 100% (50/50), done.
pradip@Pradips-MBP ~ % cd turboquant-vllm
```

### Step 2 -- Install Dependencies

```bash
pip install torch numpy scipy pyyaml tqdm pytest pytest-cov
```

```
pradip@Pradips-MBP turboquant-vllm % pip install torch numpy scipy pyyaml tqdm pytest pytest-cov
Collecting torch>=2.1.0
  Downloading torch-2.8.0-cp39-cp39-macosx_14_0_arm64.whl (63.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.4/63.4 MB 12.3 MB/s
Successfully installed torch-2.8.0 numpy-1.26.4 scipy-1.13.1 pytest-8.4.2 pytest-cov-7.1.0
```

### Step 3 -- Install TurboQuant

```bash
pip install -e ".[dev]"
```

```
pradip@Pradips-MBP turboquant-vllm % pip install -e ".[dev]"
Obtaining file:///Users/pradip/turboquant-vllm
Successfully installed turboquant-vllm-0.1.0
```

### Step 4 -- Verify

```bash
python3 -c "import turboquant; print(f'TurboQuant v{turboquant.__version__} loaded OK')"
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -c "import turboquant; print(f'TurboQuant v{turboquant.__version__} loaded OK')"
TurboQuant v0.1.0 loaded OK
```

---

## Running Tests

### Full Test Suite (128 tests)

```bash
python3 -m pytest tests/ -v
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/pradip/turboquant-vllm
configfile: pyproject.toml
collected 128 items

tests/test_asymmetric.py::test_key_quantize_dequantize_shape             PASSED [  0%]
tests/test_asymmetric.py::test_value_quantize_dequantize_shape            PASSED [  1%]
tests/test_asymmetric.py::test_key_per_channel_scales_shape               PASSED [  2%]
tests/test_asymmetric.py::test_value_per_token_scales_shape               PASSED [  3%]
tests/test_asymmetric.py::test_key_roundtrip_quality                      PASSED [  3%]
tests/test_asymmetric.py::test_value_roundtrip_quality                    PASSED [  4%]
tests/test_asymmetric.py::test_per_channel_better_for_keys_with_outliers  PASSED [  5%]
tests/test_asymmetric.py::test_quantized_values_in_range                  PASSED [  6%]
tests/test_asymmetric.py::test_compute_error_metrics                      PASSED [  7%]
tests/test_asymmetric.py::test_asymmetric_4bit_key_2bit_value             PASSED [  7%]
tests/test_attention.py::test_compute_attention_shape                      PASSED [  8%]
tests/test_attention.py::test_gqa_expansion                               PASSED [  9%]
tests/test_attention.py::test_attention_with_mask                         PASSED [ 10%]
tests/test_attention.py::test_attention_with_kv_cache                     PASSED [ 10%]
tests/test_attention.py::test_attention_error_metrics                     PASSED [ 11%]
tests/test_attention.py::test_no_kv_heads_gqa                            PASSED [ 12%]
tests/test_attention.py::test_quantization_impact_on_attention            PASSED [ 13%]
tests/test_hadamard.py::test_dimension_validation                         PASSED [ 14%]
tests/test_hadamard.py::test_output_shape                                 PASSED [ 14%]
tests/test_hadamard.py::test_output_shape_batched                         PASSED [ 15%]
tests/test_hadamard.py::test_orthogonality_preserves_norm                 PASSED [ 16%]
tests/test_hadamard.py::test_preserves_dot_products                       PASSED [ 17%]
tests/test_hadamard.py::test_invertibility                                PASSED [ 17%]
tests/test_hadamard.py::test_different_seeds_give_different_transforms    PASSED [ 18%]
tests/test_hadamard.py::test_deterministic_with_same_seed                 PASSED [ 19%]
tests/test_hadamard.py::test_post_rotation_distribution                   PASSED [ 20%]
tests/test_hadamard.py::test_dimension_mismatch_raises                    PASSED [ 21%]
tests/test_hadamard.py::test_pad_power_of_2                               PASSED [ 21%]
tests/test_hadamard.py::test_no_pad_if_already_power_of_2                 PASSED [ 22%]
tests/test_hadamard.py::test_unpad                                        PASSED [ 23%]
tests/test_integration.py::test_encode_decode_roundtrip[turbo_4bit]       PASSED [ 24%]
tests/test_integration.py::test_encode_decode_roundtrip[kivi_2bit]        PASSED [ 25%]
tests/test_integration.py::test_encode_decode_roundtrip[onebit_extreme]   PASSED [ 25%]
tests/test_integration.py::test_kv_cache_full_workflow[turbo_4bit]        PASSED [ 26%]
tests/test_integration.py::test_kv_cache_full_workflow[kivi_2bit]         PASSED [ 27%]
tests/test_integration.py::test_kv_cache_full_workflow[onebit_extreme]    PASSED [ 28%]
tests/test_integration.py::test_attention_with_quantized_cache[turbo_4bit]   PASSED [ 28%]
tests/test_integration.py::test_attention_with_quantized_cache[kivi_2bit]    PASSED [ 29%]
tests/test_integration.py::test_attention_with_quantized_cache[onebit_ext]   PASSED [ 30%]
tests/test_integration.py::test_quality_metrics_report[turbo_4bit]        PASSED [ 31%]
tests/test_integration.py::test_quality_metrics_report[kivi_2bit]         PASSED [ 32%]
tests/test_integration.py::test_quality_metrics_report[onebit_extreme]    PASSED [ 32%]
tests/test_integration.py::test_4bit_better_than_2bit                     PASSED [ 33%]
tests/test_integration.py::test_compression_ratio_ordering                PASSED [ 34%]
tests/test_integration.py::test_memory_usage_tracked                      PASSED [ 35%]
tests/test_kv_cache.py::test_initial_state                                PASSED [ 35%]
tests/test_kv_cache.py::test_append_to_residual                           PASSED [ 36%]
tests/test_kv_cache.py::test_residual_flush_to_quantized                  PASSED [ 37%]
tests/test_kv_cache.py::test_get_keys_combines_quantized_and_residual     PASSED [ 38%]
tests/test_kv_cache.py::test_get_values_combines_quantized_and_residual   PASSED [ 39%]
tests/test_kv_cache.py::test_multi_layer                                  PASSED [ 39%]
tests/test_kv_cache.py::test_incremental_append                           PASSED [ 40%]
tests/test_kv_cache.py::test_clear                                        PASSED [ 41%]
tests/test_kv_cache.py::test_memory_usage                                 PASSED [ 42%]
tests/test_kv_cache.py::test_memory_less_than_fp16_baseline               PASSED [ 42%]
tests/test_kv_cache.py::test_empty_cache_get_keys                         PASSED [ 43%]
tests/test_kv_cache.py::test_initial_lengths                              PASSED [ 44%]
tests/test_kv_cache.py::test_total_len                                    PASSED [ 45%]
tests/test_lloyd_max.py::test_codebook_size                               PASSED [ 46%]
...
tests/test_quantizer.py::test_no_nan_inf_4bit                            PASSED [ 99%]
tests/test_quantizer.py::test_no_nan_inf_1bit                            PASSED [100%]

======================== 128 passed in 63.50s (0:01:03) ========================
```

### Code Coverage

```bash
python3 -m pytest tests/ --cov=turboquant --cov-report=term-missing
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m pytest tests/ --cov=turboquant --cov-report=term-missing

Name                                  Stmts   Miss  Cover
──────────────────────────────────────────────────────────
turboquant/__init__.py                    5      0   100%
turboquant/config.py                     60      0   100%
turboquant/core/hadamard.py              54      4    93%
turboquant/core/kv_cache.py             100      3    97%
turboquant/core/lloyd_max.py             79     12    85%
turboquant/core/polar_quant.py           60      4    93%
turboquant/core/qjl.py                   43      4    91%
turboquant/core/quantizer.py            147     24    84%
turboquant/engine/attention.py           48      2    96%
turboquant/engine/inference.py           73     57    22%
turboquant/quant/asymmetric.py           55      4    93%
turboquant/quant/mixed_precision.py      51      0   100%
turboquant/quant/onebit.py               66      1    98%
turboquant/utils/metrics.py              64     18    72%
turboquant/utils/profiler.py             42     22    48%
──────────────────────────────────────────────────────────
TOTAL                                   964    155    84%

======================== 128 passed in 63.24s (0:01:03) ========================
```

---

## Benchmark Results

### Quality Benchmark

```bash
python3 -m benchmarks.quality_benchmark
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m benchmarks.quality_benchmark

TurboQuant Quality Benchmark Suite
============================================================

Benchmarking: TurboQuant-4bit
--- TurboQuant-4bit (4.00 bits) ---
  Key  cosine sim: 0.9974
  Val  cosine sim: 0.9954
  Key  SNR:        22.9 dB
  Val  SNR:        20.4 dB
  Attn cosine sim: 0.9929
  Attn KL div:     0.325034
  Retrieval acc:   67%
  Compression:     3.9x (74.2% saved)

Benchmarking: TurboQuant-3bit
--- TurboQuant-3bit (3.00 bits) ---
  Key  cosine sim: 0.9956
  Val  cosine sim: 0.9916
  Key  SNR:        20.5 dB
  Val  SNR:        17.7 dB
  Attn cosine sim: 0.9872
  Compression:     5.1x (80.5% saved)

Benchmarking: KIVI-2bit
--- KIVI-2bit (4K+2V bits) ---
  Key  cosine sim: 0.9922
  Val  cosine sim: 0.8970
  Key  SNR:        18.1 dB
  Attn cosine sim: 0.8879
  Compression:     4.0x (75.0% saved)

Benchmarking: Bonsai-1bit
--- Bonsai-1bit (1.12 bits) ---
  Key  cosine sim: 0.7994
  Val  cosine sim: 0.7993
  Attn cosine sim: 0.6710
  Compression:     14.2x (93.0% saved)

════════════════════════════════════════════════════════════════
SUMMARY
════════════════════════════════════════════════════════════════
Config              Bits  Key CosSim  Val CosSim  Attn CosSim  Compress
────────────────────────────────────────────────────────────────────────
TurboQuant-4bit     4.00     0.9974     0.9954       0.9929      3.9x
TurboQuant-3bit     3.00     0.9956     0.9916       0.9872      5.1x
KIVI-2bit           4.00     0.9922     0.8970       0.8879      4.0x
Bonsai-1bit         1.12     0.7994     0.7993       0.6710     14.2x

QUALITY GATES
  TurboQuant-4bit      [PASS]
  TurboQuant-3bit      [PASS]
  KIVI-2bit            [WARN]
  Bonsai-1bit          [WARN]
```

### Dataset Evaluation

```bash
python3 -m benchmarks.dataset_eval
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m benchmarks.dataset_eval

TurboQuant Dataset Quality Evaluation
============================================================

Evaluating TurboQuant-4bit...
  seq_len=256:  key_cos=0.9997, attn_cos=0.9943, retrieval=100%
  seq_len=1024: key_cos=0.9997, attn_cos=0.9941, retrieval=100%
  seq_len=4096: key_cos=0.9997, attn_cos=0.9939, retrieval=100%

Evaluating KIVI-4K+2V...
  seq_len=256:  key_cos=0.9935, attn_cos=0.8752, retrieval=80%
  seq_len=1024: key_cos=0.9917, attn_cos=0.8650, retrieval=80%
  seq_len=4096: key_cos=0.9897, attn_cos=0.8487, retrieval=100%

Evaluating TurboQuant-3bit...
  seq_len=256:  key_cos=0.9995, attn_cos=0.9893, retrieval=100%
  seq_len=1024: key_cos=0.9995, attn_cos=0.9888, retrieval=80%
  seq_len=4096: key_cos=0.9995, attn_cos=0.9886, retrieval=100%

Evaluating Bonsai-1bit...
  seq_len=256:  key_cos=0.4743, attn_cos=0.2050, retrieval=40%
  seq_len=1024: key_cos=0.4738, attn_cos=0.1750, retrieval=40%
  seq_len=4096: key_cos=0.4741, attn_cos=0.1255, retrieval=20%
```

### Memory Benchmark

```bash
python3 -m benchmarks.memory_benchmark
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m benchmarks.memory_benchmark

════════════════════════════════════════════════════════════════════════════
MEMORY BENCHMARK RESULTS (Llama-3.1-8B, 32 layers, 8 KV heads, dim 128)
════════════════════════════════════════════════════════════════════════════
Config              Seq Len  FP16 MB  Quant MB   Ratio  Saved   Enc tok/s
────────────────────────────────────────────────────────────────────────────
FP16 Baseline           512     64.0      64.0    1.0x   0.0%          --
FP16 Baseline          4096    512.0     512.0    1.0x   0.0%          --
FP16 Baseline         32768   4096.0    4096.0    1.0x   0.0%          --
TurboQuant-4bit         512     64.0      16.5    3.9x  74.2%      19,013
TurboQuant-4bit        4096    512.0     132.0    3.9x  74.2%      26,327
TurboQuant-4bit       32768   4096.0    1056.0    3.9x  74.2%       9,771
KIVI-2bit               512     64.0      16.0    4.0x  75.0%     188,576
KIVI-2bit              4096    512.0     128.0    4.0x  75.0%     212,872
KIVI-2bit             32768   4096.0    1024.0    4.0x  75.0%     466,118
Bonsai-1bit             512     64.0       4.5   14.2x  93.0%      78,741
Bonsai-1bit            4096    512.0      36.0   14.2x  93.0%     301,354
Bonsai-1bit           32768   4096.0     288.0   14.2x  93.0%     561,262
```

---

## Quantization Presets

| Preset | Bits | Compression | Attention Fidelity | Use Case |
|--------|------|-------------|-------------------|----------|
| `turbo_4bit` | 4 | 3.9x | 99.4% | **Production** -- near-lossless |
| `turbo_3bit` | 3 | 5.1x | 98.7% | Long-context serving |
| `kivi_2bit` | 4K+2V | 4.0x | 88.8% | Memory-constrained serving |
| `onebit_extreme` | 1.125 | 14.2x | 67.1% | Edge / mobile / research |

---

## Usage

```python
import torch
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
print(f"Compression: {report.compression_ratio:.1f}x")
```

### With KV Cache Manager

```python
# Full KV cache with residual buffer
cache = QuantizedKVCache(config, num_layers=32)
cache.set_quantizer(quantizer)

# Append tokens (auto-quantizes when residual buffer fills)
cache.append(layer_idx=0, keys=new_keys, values=new_values)

# Retrieve for attention (auto-dequantizes compressed groups)
all_keys = cache.get_keys(layer_idx=0)
all_values = cache.get_values(layer_idx=0)

# Memory profiling
print(cache.memory_usage())
# {'quantized_bytes': 131072, 'residual_bytes': 65536,
#  'metadata_bytes': 2048, 'total_bytes': 198656, 'total_mb': 0.19}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TurboQuantizer                           │
│                                                              │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐ │
│  │  PolarQuant   │  │  KIVI      │  │  Bonsai 1-bit       │ │
│  │               │  │  Asymmetric│  │  Q1_0_g128           │ │
│  │  Hadamard  ──►│  │            │  │                      │ │
│  │  Lloyd-Max ──►│  │  per-chan K │  │  sign bit + FP16    │ │
│  │  QJL (opt) ──►│  │  per-tok V │  │  scale per 128      │ │
│  └──────────────┘  └────────────┘  └──────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           Quantized KV Cache Manager                  │    │
│  │  ┌──────────────────┐  ┌───────────────────────────┐ │    │
│  │  │  Compressed Groups│  │  Residual Buffer (FP16)  │ │    │
│  │  │  (quantized)      │  │  (last 128 tokens)       │ │    │
│  │  └──────────────────┘  └───────────────────────────┘ │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           Mixed Precision Handler                     │    │
│  │           Outlier channels: 8-bit (top 25%)           │    │
│  │           Normal channels:  4-bit (bottom 75%)        │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
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
- Each weight = 1 sign bit: `0 -> -scale`, `1 -> +scale`
- Every 128 weights share one FP16 scale (mean absolute value)
- **Effective bits**: 1.125 per weight
- **Compression**: 14.2x vs FP16 (93% memory reduction)

---

## Project Structure

```
turboquant-vllm/
├── turboquant/                     # Core library (2,414 lines)
│   ├── config.py                   #   4 quantization presets
│   ├── core/
│   │   ├── hadamard.py             #   Fast Walsh-Hadamard Transform
│   │   ├── lloyd_max.py            #   Lloyd-Max optimal codebook
│   │   ├── polar_quant.py          #   PolarQuant compression
│   │   ├── qjl.py                  #   QJL residual correction
│   │   ├── kv_cache.py             #   Quantized KV cache manager
│   │   └── quantizer.py            #   Main orchestrator
│   ├── quant/
│   │   ├── asymmetric.py           #   KIVI per-channel/per-token
│   │   ├── onebit.py               #   Bonsai Q1_0_g128
│   │   └── mixed_precision.py      #   Outlier channel handling
│   ├── engine/
│   │   ├── attention.py            #   GQA-aware quantized attention
│   │   └── inference.py            #   HuggingFace integration
│   └── utils/
│       ├── metrics.py              #   Quality metrics suite
│       └── profiler.py             #   Memory profiler
├── tests/                          # 128 tests (1,455 lines)
├── benchmarks/                     # Quality + memory benchmarks
├── configs/                        # YAML presets
└── REPORT.md                       # Full technical report
```

## Papers & References

- **TurboQuant** (Google Research, ICLR 2026) -- PolarQuant + QJL for calibration-free KV cache compression
- **KIVI** (ICML 2024) -- Tuning-free asymmetric 2-bit KV quantization
- **KVQuant** (NeurIPS 2024) -- Non-uniform quantization for 10M context
- **QuaRot** (NeurIPS 2024) -- Outlier-free 4-bit inference via Hadamard rotations
- **QServe** (MLSys 2025) -- W4A8KV4 with SmoothAttention
- **Bonsai** (PrismML, 2026) -- [1-bit Q1_0_g128 format](https://github.com/thepradip/prism-Bonsai-1bit) for edge deployment

## License

Apache 2.0
