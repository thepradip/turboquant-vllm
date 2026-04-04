# TurboQuant-vLLM

### Efficient KV Cache Quantization for High-Performance LLM Inference

4-bit KV cache compression without quality loss -- built for models like [PrismML Bonsai](https://github.com/thepradip/prism-Bonsai-1bit) (1-bit weights) that already have tiny model weights but still face a **KV cache memory bottleneck** at long context.

Combines Google's TurboQuant (PolarQuant + QJL), KIVI asymmetric quantization, and Hadamard rotation to compress the KV cache that vLLM keeps in FP16.

---

## The Problem

Bonsai 1-bit models compress weights from 16 GB to 1.15 GB (14x) -- but the **KV cache stays in FP16**. At long context, the KV cache dominates memory:

```
                    Bonsai-8B at 32K context (vLLM, FP16 KV cache)
                    ──────────────────────────────────────────────
  Model weights:    █                                     1,099 MB  (1-bit Q1_0_g128)
  KV cache:         █████████████████████████              4,608 MB  (FP16 -- the bottleneck)
  Compute buffers:  ░                                       304 MB
                    ──────────────────────────────────────────────
  Total:                                                  ~6.0 GB
```

TurboQuant compresses that 4.6 GB KV cache down to **1.2 GB at 4-bit** with 99.4% attention fidelity:

```
                    Bonsai-8B at 32K context + TurboQuant 4-bit KV
                    ──────────────────────────────────────────────
  Model weights:    █                                     1,099 MB  (1-bit, unchanged)
  KV cache:         ██████                                1,187 MB  (4-bit TurboQuant)
  Compute buffers:  ░                                       304 MB
                    ──────────────────────────────────────────────
  Total:                                                  ~2.6 GB  (57% less)
```

This is what TurboQuant does: **compress the KV cache, not the weights**. Bonsai handles weights; TurboQuant handles the KV cache. Together they minimize total inference memory.

---

## How It Works

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Bonsai 1-bit Model (weights: Q1_0_g128, 1.15 GB)              │
│   ├── Embeddings, Attention, MLP -- all 1-bit                    │
│   └── KV cache generated during inference (FP16 by default)     │
│                                                                   │
│   TurboQuant (this project) -- compresses the KV cache           │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │                                                           │  │
│   │  ┌──────────────┐  ┌────────────┐                        │  │
│   │  │  PolarQuant   │  │  KIVI      │   Quantization        │  │
│   │  │  Hadamard  ──►│  │  Asymmetric│   Strategies           │  │
│   │  │  Lloyd-Max ──►│  │  per-chan K │                        │  │
│   │  │  QJL (opt) ──►│  │  per-tok V │                        │  │
│   │  └──────────────┘  └────────────┘                        │  │
│   │                                                           │  │
│   │  ┌─────────────────────────────────────────────────────┐ │  │
│   │  │         Quantized KV Cache Manager                   │ │  │
│   │  │  ┌──────────────────┐  ┌──────────────────────────┐ │ │  │
│   │  │  │  Compressed Cache │  │  Residual Buffer (FP16) │ │ │  │
│   │  │  │  (4-bit groups)   │  │  (last 128 tokens)      │ │ │  │
│   │  │  └──────────────────┘  └──────────────────────────┘ │ │  │
│   │  └─────────────────────────────────────────────────────┘ │  │
│   │                                                           │  │
│   │  Mixed Precision: outlier channels 8-bit, normal 4-bit   │  │
│   └───────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Key distinction:**
- **Bonsai Q1_0_g128** = 1-bit **weight** quantization (model parameters). KV cache stays FP16.
- **TurboQuant** = 4-bit **KV cache** quantization (runtime activations). Weights are untouched.
- **Together** = 1-bit weights + 4-bit KV cache = maximum memory efficiency.

---

## KV Cache Quantization Presets

| Preset | KV Bits | KV Compression | Attention Fidelity | Use Case |
|--------|---------|----------------|-------------------|----------|
| `turbo_4bit` | 4 | 3.9x | 99.4% | **Production** -- near-lossless, pairs with Bonsai |
| `turbo_3bit` | 3 | 5.1x | 98.7% | Long-context serving |
| `kivi_2bit` | 4K+2V | 4.0x | 88.8% | Memory-constrained |
| `onebit_extreme` | 1.125 | 14.2x | 81.0% | **Research** -- Hadamard-rotated 1-bit KV cache |

The `onebit_extreme` preset applies Hadamard rotation before 1-bit sign quantization (TurboQuant + Bonsai hybrid for KV cache). This is experimental -- for production use `turbo_4bit`.

---

## Total Memory Savings (Bonsai-8B + TurboQuant)

| Context | Bonsai + FP16 KV | Bonsai + TurboQuant 4-bit KV | Saved |
|---------|-------------------|-------------------------------|-------|
| 4K | 1.67 GB | 1.25 GB | 25% |
| 16K | 3.40 GB | 1.83 GB | 46% |
| 32K | 5.71 GB | 2.59 GB | **55%** |
| 64K | 10.3 GB | 3.91 GB | **62%** |
| 128K | 19.5 GB | 6.55 GB | **66%** |

At 128K context, TurboQuant saves ~13 GB by compressing the KV cache alone.

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

### Full Test Suite (153 tests)

```bash
python3 -m pytest tests/ -v
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0
collected 153 items

tests/test_asymmetric.py       10 passed
tests/test_attention.py         7 passed
tests/test_hadamard.py         14 passed
tests/test_integration.py      15 passed
tests/test_kv_cache.py         13 passed
tests/test_lloyd_max.py        11 passed
tests/test_metrics.py          12 passed
tests/test_mixed_precision.py   6 passed
tests/test_onebit.py           13 passed
tests/test_polar_quant.py      13 passed
tests/test_profiler.py          9 passed
tests/test_qjl.py             10 passed
tests/test_quantizer.py        20 passed

======================== 153 passed in 67.09s (0:01:07) ========================
```

### Code Coverage (90%)

```bash
python3 -m pytest tests/ --cov=turboquant --cov-report=term-missing
```

```
Name                                  Stmts   Miss  Cover
──────────────────────────────────────────────────────────
turboquant/config.py                     60      0   100%
turboquant/core/hadamard.py              54      1    98%
turboquant/core/kv_cache.py             100      3    97%
turboquant/core/polar_quant.py           60      0   100%
turboquant/core/quantizer.py            147      8    95%
turboquant/engine/attention.py           48      2    96%
turboquant/quant/asymmetric.py           55      4    93%
turboquant/quant/mixed_precision.py      51      0   100%
turboquant/quant/onebit.py               66      1    98%
turboquant/utils/metrics.py              64      0   100%
turboquant/utils/profiler.py             42      5    88%
──────────────────────────────────────────────────────────
TOTAL                                   964     93    90%
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

--- TurboQuant-4bit (4.00 bits) ---
  Key  cosine sim: 0.9974
  Val  cosine sim: 0.9954
  Key  SNR:        22.9 dB
  Attn cosine sim: 0.9929
  Compression:     3.9x (74.2% saved)

--- TurboQuant-3bit (3.00 bits) ---
  Key  cosine sim: 0.9956
  Val  cosine sim: 0.9916
  Key  SNR:        20.5 dB
  Attn cosine sim: 0.9872
  Compression:     5.1x (80.5% saved)

--- KIVI-2bit (4K+2V bits) ---
  Key  cosine sim: 0.9922
  Val  cosine sim: 0.8970
  Attn cosine sim: 0.8879
  Compression:     4.0x (75.0% saved)

QUALITY GATES
  TurboQuant-4bit      [PASS]
  TurboQuant-3bit      [PASS]
  KIVI-2bit            [WARN]
```

### Memory Benchmark (KV Cache Only)

```bash
python3 -m benchmarks.memory_benchmark
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m benchmarks.memory_benchmark

Config              Seq Len  FP16 KV   Quant KV   Ratio  Saved   Enc tok/s
────────────────────────────────────────────────────────────────────────────
FP16 Baseline          4096    512 MB    512 MB    1.0x   0.0%          --
FP16 Baseline         32768   4096 MB   4096 MB    1.0x   0.0%          --
TurboQuant-4bit        4096    512 MB    132 MB    3.9x  74.2%      26,327
TurboQuant-4bit       32768   4096 MB   1056 MB    3.9x  74.2%       9,771
KIVI-2bit              4096    512 MB    128 MB    4.0x  75.0%     212,872
KIVI-2bit             32768   4096 MB   1024 MB    4.0x  75.0%     466,118
```

### Dataset Evaluation

```bash
python3 -m benchmarks.dataset_eval
```

```
pradip@Pradips-MBP turboquant-vllm % python3 -m benchmarks.dataset_eval

Evaluating TurboQuant-4bit...
  seq_len=256:  key_cos=0.9997, attn_cos=0.9943, retrieval=100%
  seq_len=1024: key_cos=0.9997, attn_cos=0.9941, retrieval=100%
  seq_len=4096: key_cos=0.9997, attn_cos=0.9939, retrieval=100%

Evaluating TurboQuant-3bit...
  seq_len=256:  key_cos=0.9995, attn_cos=0.9893, retrieval=100%
  seq_len=1024: key_cos=0.9995, attn_cos=0.9888, retrieval=80%
  seq_len=4096: key_cos=0.9995, attn_cos=0.9886, retrieval=100%
```

---

## Usage

```python
import torch
from turboquant import TurboQuantConfig, TurboQuantizer, QuantizedKVCache

# 4-bit KV cache quantization (recommended for production with Bonsai models)
config = TurboQuantConfig.turbo_4bit(
    num_heads=32, num_kv_heads=8, head_dim=128
)

quantizer = TurboQuantizer(config)

# These are the KV cache tensors generated during inference (normally FP16)
keys = torch.randn(1, 512, 8, 128)    # (batch, seq, kv_heads, dim)
values = torch.randn(1, 512, 8, 128)

# Compress KV cache: FP16 -> 4-bit
q_keys, key_meta = quantizer.encode_keys(keys)
q_vals, val_meta = quantizer.encode_values(values)

# Decompress for attention computation
recon_keys = quantizer.decode_keys(q_keys, key_meta)
recon_vals = quantizer.decode_values(q_vals, val_meta)

# Verify quality
from turboquant.utils.metrics import QualityMetrics
report = QualityMetrics.full_report(keys, recon_keys, effective_bits=4.0)
print(f"Cosine similarity: {report.cosine_similarity_mean:.4f}")  # 0.9975
print(f"SNR: {report.snr_db:.1f} dB")                            # 22.9 dB
print(f"Compression: {report.compression_ratio:.1f}x")            # 4.0x
```

### With KV Cache Manager

```python
# Streaming KV cache with auto-quantization and residual buffer
cache = QuantizedKVCache(config, num_layers=32)
cache.set_quantizer(quantizer)

# Append tokens -- recent tokens stay FP16, older tokens auto-quantized to 4-bit
cache.append(layer_idx=0, keys=new_keys, values=new_values)

# Retrieve for attention -- auto-dequantizes compressed groups
all_keys = cache.get_keys(layer_idx=0)
all_values = cache.get_values(layer_idx=0)

print(cache.memory_usage())
```

---

## Algorithm Details

### PolarQuant (TurboQuant Primary Stage) -- for KV cache compression
1. **Decompose** each KV vector into magnitude `r` (FP16 scalar) and direction `d` (unit vector)
2. **Rotate** direction with randomized Hadamard transform -- redistributes outliers uniformly
3. **Quantize** rotated coordinates with fixed Lloyd-Max codebook (no per-block metadata needed)
4. **Pack** quantized indices into sub-byte storage

### KIVI Asymmetric Quantization -- for KV cache compression
- **Keys: per-channel** -- outlier channels are persistent across tokens, so quantize along token dimension per channel
- **Values: per-token** -- no channel outlier pattern; attention is sparse, so quantize each token independently
- **Result**: 5x lower attention score error for keys, 15x lower for values vs. naive approaches

### Bonsai Q1_0_g128 -- for model weight compression (separate concern)
- Bonsai quantizes **model weights** to 1-bit, not KV cache
- Each weight = 1 sign bit: `0 -> -scale`, `1 -> +scale`
- Every 128 weights share one FP16 scale
- KV cache generated by Bonsai models is still FP16 -- **that's what TurboQuant compresses**

---

## Project Structure

```
turboquant-vllm/
├── turboquant/                     # Core library (2,414 lines)
│   ├── config.py                   #   4 KV cache quantization presets
│   ├── core/
│   │   ├── hadamard.py             #   Fast Walsh-Hadamard Transform
│   │   ├── lloyd_max.py            #   Lloyd-Max optimal codebook
│   │   ├── polar_quant.py          #   PolarQuant compression
│   │   ├── qjl.py                  #   QJL residual correction
│   │   ├── kv_cache.py             #   Quantized KV cache manager
│   │   └── quantizer.py            #   Main orchestrator
│   ├── quant/
│   │   ├── asymmetric.py           #   KIVI per-channel/per-token
│   │   ├── onebit.py               #   1-bit KV (Hadamard + sign-bit, experimental)
│   │   └── mixed_precision.py      #   Outlier channel handling
│   ├── engine/
│   │   ├── attention.py            #   GQA-aware quantized attention
│   │   └── inference.py            #   HuggingFace integration
│   └── utils/
│       ├── metrics.py              #   Quality metrics suite
│       └── profiler.py             #   Memory profiler
├── tests/                          # 153 tests, 90% coverage
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
- **Bonsai** (PrismML, 2026) -- [1-bit weight quantization](https://github.com/thepradip/prism-Bonsai-1bit) (model weights, not KV cache)

## License

Apache 2.0
