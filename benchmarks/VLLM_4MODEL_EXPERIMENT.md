# vLLM Four-Model TurboQuant KV Experiment

This experiment is for the standard vLLM path where we can access the
attention/KV stack. Use GGUF as a separate deployment comparison, not as the
primary TurboQuant KV integration path.

## Models

Edit `benchmarks/vllm_4model_models.example.jsonc` to point to the exact 4-bit
AWQ or GPTQ repositories available in your environment:

- Qwen 3.5 4B
- Qwen 3.5 9B
- Gemma 4 E4B IT
- Llama 3.1 8B Instruct

If a model is not actually published under the default ID in the example file,
replace `model`, `tokenizer`, and `weight_quantization` with the real repo and
vLLM quantization mode.

## Metrics

The runner writes JSONL records with:

- quality pass/fail against the 54 reliable questions from the TurboQuant MLX
  benchmark set
- `wall_time_s`
- `ttft_ms` when vLLM exposes first-token metrics
- prompt and generated token counts
- decode throughput proxy
- external GPU memory peak
- baseline-subtracted observed request memory
- sampled GPU utilization and GPU memory-controller utilization
- observed memory saved vs fp16 KV in the summary

External GPU memory is preferred over allocator-only counters. The runner uses
`pynvml` when available, then falls back to `nvidia-smi`.

On T4 hosts without `nvcc`, force Triton attention to avoid FlashInfer JIT:

```bash
--attention-backend TRITON_ATTN
```

The runner now does this automatically when `nvcc` is absent. For larger
4-bit models on a 15 GB T4, also use eager mode or a smaller KV cache if CUDA
graph profiling runs out of memory:

```bash
--enforce-eager --gpu-memory-utilization 0.82
```

## Smoke Test

Run one question at one context before a full sweep:

```bash
python3 benchmarks/vllm_4model_experiment.py \
  --models-config benchmarks/vllm_4model_models.example.jsonc \
  --questions /tmp/turboquant-mlx/benchmarks/tq_eval_65_questions.json \
  --contexts 2048 \
  --kv-configs fp16 turboquant_4bit \
  --max-questions 1 \
  --repeats 1 \
  --enforce-eager \
  --output results/vllm_4model_smoke.jsonl
```

## Full Run

```bash
python3 benchmarks/vllm_4model_experiment.py \
  --models-config benchmarks/vllm_4model_models.example.jsonc \
  --questions /tmp/turboquant-mlx/benchmarks/tq_eval_65_questions.json \
  --contexts 2048 4096 8192 16384 32768 \
  --kv-configs fp16 fp8 turboquant_4bit turboquant_3bit \
  --repeats 3 \
  --max-tokens 512 \
  --output results/vllm_4model_full.jsonl
```

## Summarize Existing Results

```bash
python3 benchmarks/vllm_4model_experiment.py \
  --output results/vllm_4model_full.jsonl \
  --summarize-only
```

## Runtime Requirements

- CUDA GPU
- `torch`
- `transformers`
- `vllm`
- this package installed so the TurboQuant vLLM plugin entry point is visible
- `pynvml` or `nvidia-smi` for unbiased external GPU memory sampling
