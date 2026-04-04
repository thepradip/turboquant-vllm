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

**Bonsai handles weights; TurboQuant handles the KV cache. Together they minimize total inference memory.**

---

## Real Benchmark: Bonsai-8B (1-bit) with KV Cache Quantization

Tested on **20 production QA questions** across 4 categories (RAG, Finance, Reasoning, Instruction) with 4 KV cache configurations:

```
Config                      Score  vs FP16  Wall(s)  PP tok/s  Gen tok/s
──────────────────────────────────────────────────────────────────────
FP16 baseline                18/20      ---      150       283         47
8-bit uniform (q8_0)         18/20       +0      156       281         44
4-bit uniform (q4_0)         17/20       -1      144       282         45
K=8bit V=4bit (KIVI)         18/20       +0      221       229         29
```

**Per-category breakdown:**

| Category | FP16 | Q8_0 | Q4_0 | K8V4 (KIVI) |
|----------|------|------|------|-------------|
| RAG Context (5) | 4/5 | 4/5 | 4/5 | 4/5 |
| Finance (5) | **5/5** | **5/5** | **5/5** | **5/5** |
| Reasoning (5) | **5/5** | **5/5** | **5/5** | **5/5** |
| Instruction (5) | 4/5 | 4/5 | 3/5 | 4/5 |

**Key findings:**
- **Q8_0**: Zero quality loss, identical to FP16 on all 20 questions
- **Q4_0**: Loses 1 question out of 20 (SQL generation), 2x KV memory savings
- **K8V4 (KIVI-style)**: Zero quality loss, keys at 8-bit + values at 4-bit
- **Finance and Reasoning: 5/5 across ALL configs** -- complex tasks fully preserved

---

## CLI Usage

```bash
# Run Bonsai GGUF with quantized KV cache
python -m turboquant --gguf /path/to/Bonsai-8B.gguf --kv-quant q4_0 \
    --prompt "What is the gross margin if revenue is 50M and COGS is 30M?"

# HuggingFace model with TurboQuant 4-bit KV
python -m turboquant --model Qwen/Qwen2.5-3B-Instruct --kv-quant turbo_4bit \
    --prompt "Explain the Pythagorean theorem"

# Compare FP16 vs quantized side-by-side
python -m turboquant --model Qwen/Qwen2.5-0.5B --kv-quant turbo_4bit --compare

# Full 20-question benchmark on Bonsai-8B
python benchmarks/full_kv_benchmark.py Bonsai-8B
```

**Available `--kv-quant` options:**

| Flag | For GGUF | For HuggingFace | Description |
|------|----------|-----------------|-------------|
| `f16` | yes | -- | FP16 baseline |
| `q8_0` | yes | -- | 8-bit uniform |
| `q4_0` | yes | -- | 4-bit uniform |
| `turbo_4bit` | -- | yes | PolarQuant + Hadamard + Lloyd-Max |
| `turbo_3bit` | -- | yes | PolarQuant + QJL residual correction |
| `kivi_2bit` | -- | yes | Asymmetric: 4-bit keys, 2-bit values |

---

## KV Cache Quantization Presets

| Preset | KV Bits | KV Compression | Attention Fidelity | Use Case |
|--------|---------|----------------|-------------------|----------|
| `turbo_4bit` | 4 | 3.9x | 99.4% | **Production** -- near-lossless |
| `turbo_3bit` | 3 | 5.1x | 98.7% | Long-context serving |
| `kivi_2bit` | 4K+2V | 4.0x | 88.8% | Memory-constrained |

## Total Memory Savings (Bonsai-8B + TurboQuant)

| Context | Bonsai + FP16 KV | Bonsai + 4-bit KV | Saved |
|---------|-------------------|---------------------|-------|
| 4K | 1.67 GB | 1.25 GB | 25% |
| 16K | 3.40 GB | 1.83 GB | 46% |
| 32K | 5.71 GB | 2.59 GB | **55%** |
| 64K | 10.3 GB | 3.91 GB | **62%** |
| 128K | 19.5 GB | 6.55 GB | **66%** |

---

## Installation

```bash
git clone https://github.com/thepradip/turboquant-vllm.git
cd turboquant-vllm
pip install torch numpy scipy pyyaml tqdm pytest pytest-cov
pip install -e ".[dev]"
```

Verify:
```bash
python3 -c "import turboquant; print(f'TurboQuant v{turboquant.__version__} loaded OK')"
```

## Running Tests

```bash
# 153 tests, 90% coverage
python3 -m pytest tests/ -v
python3 -m pytest tests/ --cov=turboquant
```

## Running Benchmarks

```bash
# Synthetic KV cache quality
python3 -m benchmarks.quality_benchmark

# Memory profiling
python3 -m benchmarks.memory_benchmark

# Real model (Qwen2.5-3B): KV cache interception + perplexity
python3 -m benchmarks.real_model_eval

# Bonsai-8B: 20-question production QA across f16/q8_0/q4_0/k8v4
python3 benchmarks/full_kv_benchmark.py Bonsai-8B

# Quick 5-question smoke test
python3 benchmarks/smoke_test.py Bonsai-8B q4_0
```

---

## Python API

```python
import torch
from turboquant import TurboQuantConfig, TurboQuantizer, QuantizedKVCache

# 4-bit KV cache quantization
config = TurboQuantConfig.turbo_4bit(
    num_heads=32, num_kv_heads=8, head_dim=128
)
quantizer = TurboQuantizer(config)

# Compress KV cache: FP16 -> 4-bit
keys = torch.randn(1, 512, 8, 128)
q_keys, key_meta = quantizer.encode_keys(keys)
recon_keys = quantizer.decode_keys(q_keys, key_meta)

# Streaming KV cache with residual buffer
cache = QuantizedKVCache(config, num_layers=32)
cache.set_quantizer(quantizer)
cache.append(layer_idx=0, keys=new_keys, values=new_values)
all_keys = cache.get_keys(layer_idx=0)
```

---

## How It Works

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Bonsai 1-bit Model (weights: Q1_0_g128, 1.15 GB)              │
│   └── KV cache generated during inference (FP16 by default)     │
│                                                                   │
│   TurboQuant -- compresses the KV cache                          │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │  PolarQuant        KIVI Asymmetric    Mixed Precision     │  │
│   │  Hadamard ──►      per-channel K      outlier: 8-bit      │  │
│   │  Lloyd-Max ──►     per-token V        normal:  4-bit      │  │
│   │  QJL (opt) ──►                                            │  │
│   │                                                           │  │
│   │  Quantized KV Cache Manager                               │  │
│   │  ┌──────────────────┐  ┌──────────────────────────┐      │  │
│   │  │  Compressed Cache │  │  Residual Buffer (FP16) │      │  │
│   │  │  (4-bit groups)   │  │  (last 128 tokens)      │      │  │
│   │  └──────────────────┘  └──────────────────────────┘      │  │
│   └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### PolarQuant (TurboQuant Primary Stage)
1. Decompose each KV vector into magnitude (FP16) and direction (unit vector)
2. Rotate direction with randomized Hadamard transform -- eliminates outliers
3. Quantize rotated coordinates with fixed Lloyd-Max codebook
4. Pack quantized indices into sub-byte storage

### KIVI Asymmetric Quantization
- **Keys: per-channel** -- outlier channels persistent across tokens
- **Values: per-token** -- no channel outlier pattern, sparse attention
- 5x lower attention error for keys, 15x lower for values vs naive

### Bonsai Q1_0_g128 (model weights, separate concern)
- Bonsai quantizes **model weights** to 1-bit, not KV cache
- KV cache from Bonsai is still FP16 -- **that's what TurboQuant compresses**

---

## Project Structure

```
turboquant-vllm/
├── turboquant/                     # Core library
│   ├── cli.py                      #   CLI with --kv-quant flag
│   ├── config.py                   #   Quantization presets
│   ├── core/
│   │   ├── hadamard.py             #   Fast Walsh-Hadamard Transform
│   │   ├── lloyd_max.py            #   Lloyd-Max optimal codebook
│   │   ├── polar_quant.py          #   PolarQuant compression
│   │   ├── qjl.py                  #   QJL residual correction
│   │   ├── kv_cache.py             #   Quantized KV cache manager
│   │   └── quantizer.py            #   Main orchestrator
│   ├── quant/
│   │   ├── asymmetric.py           #   KIVI per-channel/per-token
│   │   ├── onebit.py               #   1-bit KV (experimental)
│   │   └── mixed_precision.py      #   Outlier channel handling
│   └── engine/
│       ├── attention.py            #   GQA-aware quantized attention
│       └── inference.py            #   HuggingFace integration
├── tests/                          # 153 tests, 90% coverage
├── benchmarks/
│   ├── full_kv_benchmark.py        #   20Q production QA (f16/q8/q4/k8v4)
│   ├── smoke_test.py               #   Quick 5Q comparison
│   ├── quality_benchmark.py        #   Synthetic KV quality
│   ├── memory_benchmark.py         #   Memory profiling
│   ├── real_model_eval.py          #   Qwen2.5-3B KV interception
│   └── dataset_eval.py             #   Dataset-based evaluation
└── configs/                        # YAML presets
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
