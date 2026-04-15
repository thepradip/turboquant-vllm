#!/usr/bin/env python3
"""Four-model vLLM experiment for TurboQuant KV cache evaluation.

This runner is intentionally focused on the standard vLLM path, not GGUF.
GGUF is useful as a deployment comparison, but KV cache integration work is
cleaner when vLLM owns the attention/KV stack directly.

It records one JSONL row per model/context/KV-config/question/repeat with:
- quality checks from the TurboQuant MLX question set
- wall time, TTFT proxy, prompt/decode throughput when vLLM exposes it
- unbiased observed GPU memory using baseline-subtracted external telemetry

Example:
  python3 benchmarks/vllm_4model_experiment.py \
    --questions /tmp/turboquant-mlx/benchmarks/tq_eval_65_questions.json \
    --contexts 2048 4096 8192 16384 32768 \
    --kv-configs fp16 fp8 turboquant_4bit turboquant_3bit \
    --output results/vllm_4model.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_MODELS = [
    {
        "label": "qwen3.5-4b",
        "model": "Qwen/Qwen3.5-4B-Instruct",
        "tokenizer": "Qwen/Qwen3.5-4B-Instruct",
        "weight_quantization": "awq",
    },
    {
        "label": "qwen3.5-9b",
        "model": "Qwen/Qwen3.5-9B-Instruct",
        "tokenizer": "Qwen/Qwen3.5-9B-Instruct",
        "weight_quantization": "awq",
    },
    {
        "label": "gemma-4-e4b-it",
        "model": "google/gemma-4-E4B-it",
        "tokenizer": "google/gemma-4-E4B-it",
        "weight_quantization": "awq",
    },
    {
        "label": "llama-3.1-8b-instruct",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer": "meta-llama/Llama-3.1-8B-Instruct",
        "weight_quantization": "awq",
    },
]


KV_CONFIGS = {
    "fp16": {"kv_cache_dtype": "auto", "turboquant_bits": None},
    "fp8": {"kv_cache_dtype": "fp8", "turboquant_bits": None},
    "turboquant_4bit": {"kv_cache_dtype": "turboquant_4bit", "turboquant_bits": 4},
    "turboquant_3bit": {"kv_cache_dtype": "turboquant_3bit", "turboquant_bits": 3},
}


def vllm_effective_kv_storage_dtype(kv_name: str) -> str:
    kv_dtype = KV_CONFIGS[kv_name]["kv_cache_dtype"]
    if str(kv_dtype).startswith("turboquant"):
        return "auto"
    return str(kv_dtype)


@dataclass
class GpuSnapshot:
    timestamp_s: float
    used_mb: int
    gpu_util_pct: int | None = None
    memory_util_pct: int | None = None


class GpuMemorySampler:
    """External GPU memory sampler.

    Prefer pynvml if installed. Fall back to nvidia-smi. If neither exists,
    the experiment still runs and marks memory fields as null.
    """

    def __init__(self, interval_s: float = 0.1, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.samples: list[GpuSnapshot] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self._nvidia_smi = shutil.which("nvidia-smi")

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self._nvml = None
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None or self._nvidia_smi is not None

    def read_used_mb(self) -> int | None:
        sample = self.read_sample()
        return sample.used_mb if sample is not None else None

    def read_sample(self) -> GpuSnapshot | None:
        if self._handle is not None and self._nvml is not None:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            return GpuSnapshot(
                timestamp_s=time.time(),
                used_mb=int(info.used / (1024 * 1024)),
                gpu_util_pct=int(util.gpu),
                memory_util_pct=int(util.memory),
            )

        if self._nvidia_smi is None:
            return None

        try:
            out = subprocess.check_output(
                [
                    self._nvidia_smi,
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            used_mb, gpu_util, memory_util = [
                value.strip() for value in out.strip().splitlines()[0].split(",")
            ]
            return GpuSnapshot(
                timestamp_s=time.time(),
                used_mb=int(used_mb),
                gpu_util_pct=int(gpu_util),
                memory_util_pct=int(memory_util),
            )
        except Exception:
            return None

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self.read_sample()
            if sample is not None:
                self.samples.append(sample)
            time.sleep(self.interval_s)

    def peak_mb(self) -> int | None:
        if not self.samples:
            return None
        return max(s.used_mb for s in self.samples)

    def peak_gpu_util_pct(self) -> int | None:
        values = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        return max(values) if values else None

    def mean_gpu_util_pct(self) -> float | None:
        values = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        return round(mean(values), 2) if values else None

    def peak_memory_util_pct(self) -> int | None:
        values = [
            s.memory_util_pct for s in self.samples if s.memory_util_pct is not None
        ]
        return max(values) if values else None

    def mean_memory_util_pct(self) -> float | None:
        values = [
            s.memory_util_pct for s in self.samples if s.memory_util_pct is not None
        ]
        return round(mean(values), 2) if values else None


def load_questions(path: Path, max_questions: int | None) -> list[dict[str, Any]]:
    with path.open() as f:
        raw = json.load(f)
    questions = [q for q in raw["questions"] if q.get("reliable", True)]
    if max_questions is not None:
        questions = questions[:max_questions]
    return questions


def filter_questions_by_id(
    questions: list[dict[str, Any]], question_ids: list[str] | None
) -> list[dict[str, Any]]:
    if not question_ids:
        return questions

    requested = set(question_ids)
    filtered = [q for q in questions if q["id"] in requested]
    found = {q["id"] for q in filtered}
    missing = sorted(requested - found)
    if missing:
        raise SystemExit(f"Unknown --question-ids entries: {', '.join(missing)}")
    return filtered


def find_default_questions() -> Path | None:
    candidates = [
        Path("benchmarks/tq_eval_65_questions.json"),
        Path("/tmp/turboquant-mlx/benchmarks/tq_eval_65_questions.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_model_matrix(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_MODELS
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data["models"]
    return data


def filter_model_matrix(
    models: list[dict[str, Any]], labels: list[str] | None
) -> list[dict[str, Any]]:
    if not labels:
        return models

    requested = set(labels)
    filtered = [
        model
        for model in models
        if model.get("label", model.get("model")) in requested
    ]
    found = {model.get("label", model.get("model")) for model in filtered}
    missing = sorted(requested - found)
    if missing:
        raise SystemExit(f"Unknown --model-labels entries: {', '.join(missing)}")
    return filtered


def load_completed_keys(path: Path) -> set[tuple[str, int, str, int, str]]:
    """Return per-question rows already written to a JSONL output file."""
    completed = set()
    if not path.exists():
        return completed

    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "repeat" not in row or "question_id" not in row:
                continue
            completed.add(
                (
                    row["model_label"],
                    int(row["context"]),
                    row["kv_config"],
                    int(row["repeat"]),
                    row["question_id"],
                )
            )
    return completed


def check_answer(answer: str, question: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    text = answer.lower()
    details: dict[str, Any] = {"numbers": {}, "words": {}, "order": []}
    passed = True

    for number in question.get("check_numbers", []):
        found = str(number).lower() in text
        details["numbers"][number] = found
        passed = passed and found

    for word in question.get("check_words", []):
        found = word.lower() in text
        details["words"][word] = found
        passed = passed and found

    order_terms = [term.lower() for term in question.get("check_order", [])]
    if order_terms:
        positions = [text.find(term) for term in order_terms]
        details["order"] = positions
        order_ok = all(pos >= 0 for pos in positions) and positions == sorted(positions)
        passed = passed and order_ok

    return passed, details


def make_prompt(tokenizer: Any, raw_prompt: str) -> str:
    messages = [{"role": "user", "content": raw_prompt}]
    try:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        return raw_prompt


def pad_to_context(
    tokenizer: Any, prompt: str, target_context: int, max_new_tokens: int
) -> str:
    """Pad prompt with deterministic filler to approximate target context."""
    try:
        n_tokens = len(tokenizer.encode(prompt))
    except Exception:
        return prompt

    target_prompt_tokens = max(1, target_context - max_new_tokens - 8)
    if n_tokens >= target_prompt_tokens:
        return prompt

    filler = (
        "\nContext filler: This paragraph is deterministic neutral text used "
        "only to occupy context tokens for latency and memory evaluation. "
        "Ignore it unless the question explicitly asks about it.\n"
    )
    pieces = []
    while n_tokens < target_prompt_tokens:
        candidate = "".join(pieces) + filler + prompt
        try:
            candidate_tokens = len(tokenizer.encode(candidate))
        except Exception:
            break
        if candidate_tokens > target_prompt_tokens:
            break
        pieces.append(filler)
        n_tokens = candidate_tokens
    return "".join(pieces) + prompt


def request_metrics(output: Any, wall_s: float) -> dict[str, Any]:
    metrics = getattr(output, "metrics", None)
    prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
    generated_token_ids = getattr(output.outputs[0], "token_ids", []) if output.outputs else []
    gen_tokens = len(generated_token_ids or [])

    row = {
        "wall_time_s": round(wall_s, 6),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "decode_tps": round(gen_tokens / wall_s, 4) if wall_s > 0 else None,
        "ttft_ms": None,
        "time_in_queue_s": None,
        "prefill_time_s": None,
        "decode_time_s": None,
    }

    if metrics is not None:
        first_token_time = getattr(metrics, "first_token_time", None)
        arrival_time = getattr(metrics, "arrival_time", None)
        if first_token_time is not None and arrival_time is not None:
            row["ttft_ms"] = round((first_token_time - arrival_time) * 1000, 3)
        for attr in ("time_in_queue", "prefill_time", "decode_time"):
            val = getattr(metrics, attr, None)
            if val is not None:
                row[f"{attr}_s"] = round(float(val), 6)

    return row


def build_llm(model_cfg: dict[str, Any], kv_name: str, context: int, args: argparse.Namespace) -> Any:
    from vllm import LLM  # type: ignore

    kv = KV_CONFIGS[kv_name]
    kwargs: dict[str, Any] = {
        "model": model_cfg["model"],
        "tokenizer": model_cfg.get("tokenizer", model_cfg["model"]),
        "max_model_len": context,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": kv["kv_cache_dtype"],
    }

    weight_quant = model_cfg.get("weight_quantization")
    if weight_quant and weight_quant not in {"none", "auto"}:
        kwargs["quantization"] = weight_quant

    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend

    if args.enforce_eager:
        kwargs["enforce_eager"] = True

    if args.kv_cache_memory_bytes is not None:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes

    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    tensor_parallel_size = int(model_cfg.get("tensor_parallel_size", args.tensor_parallel_size))
    kwargs["tensor_parallel_size"] = tensor_parallel_size

    extra = model_cfg.get("vllm_kwargs", {})
    kwargs.update(extra)
    return LLM(**kwargs)


def shutdown_llm(llm: Any) -> None:
    """Best-effort cleanup so sequential model loads do not keep vLLM workers alive."""
    if llm is None:
        return

    try:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown(timeout=30)
    except Exception as exc:
        print(f"Warning: vLLM engine shutdown failed: {exc}", flush=True)

    del llm
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_suite(args: argparse.Namespace) -> None:
    try:
        from vllm import SamplingParams  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        import turboquant.vllm_plugin as tq_plugin

        tq_plugin.register()
    except Exception as exc:
        raise SystemExit(
            "Missing runtime dependency for the 4-model experiment. "
            "Install torch, transformers, vllm, and this package with the vllm extra. "
            f"Original error: {exc}"
        ) from exc

    question_path = args.questions or find_default_questions()
    if question_path is None:
        raise SystemExit(
            "Question file not found. Pass --questions pointing to "
            "turboquant-mlx/benchmarks/tq_eval_65_questions.json."
        )

    questions = filter_questions_by_id(
        load_questions(question_path, args.max_questions), args.question_ids
    )
    models = filter_model_matrix(load_model_matrix(args.models_config), args.model_labels)
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(out_path) if args.resume_skip_existing else set()

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    with out_path.open("a", buffering=1) as out:
        for model_cfg in models:
            tokenizer = AutoTokenizer.from_pretrained(
                model_cfg.get("tokenizer", model_cfg["model"]),
                trust_remote_code=args.trust_remote_code,
            )

            for context in args.contexts:
                for kv_name in args.kv_configs:
                    model_label = model_cfg.get("label", model_cfg["model"])
                    has_remaining = any(
                        (
                            model_label,
                            context,
                            kv_name,
                            repeat,
                            q["id"],
                        )
                        not in completed
                        for repeat in range(args.repeats)
                        for q in questions
                    )
                    if not has_remaining:
                        continue

                    sampler = GpuMemorySampler(args.memory_sample_interval_s, args.gpu_index)
                    idle_mb_before_load = sampler.read_used_mb()
                    load_started = time.perf_counter()
                    status = "ok"
                    error = None
                    llm = None
                    try:
                        llm = build_llm(model_cfg, kv_name, context, args)
                        load_wall_s = time.perf_counter() - load_started
                        idle_loaded_mb = sampler.read_used_mb()

                        prompts = []
                        prompt_questions = []
                        for q in questions:
                            raw_prompt = q.get("prompt") or q["question"]
                            formatted = make_prompt(tokenizer, raw_prompt)
                            prompts.append(
                                pad_to_context(
                                    tokenizer, formatted, context, args.max_tokens
                                )
                            )
                            prompt_questions.append(q)

                        for repeat in range(args.repeats):
                            for q, prompt in zip(prompt_questions, prompts):
                                row_key = (
                                    model_label,
                                    context,
                                    kv_name,
                                    repeat,
                                    q["id"],
                                )
                                if row_key in completed:
                                    continue

                                sampler.start()
                                t0 = time.perf_counter()
                                try:
                                    outputs = llm.generate([prompt], sampling)
                                    wall_s = time.perf_counter() - t0
                                    answer = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
                                    passed, check_details = check_answer(answer, q)
                                    req_metrics = request_metrics(outputs[0], wall_s)
                                    req_status = "ok"
                                    req_error = None
                                except Exception as exc:
                                    wall_s = time.perf_counter() - t0
                                    answer = ""
                                    passed = False
                                    check_details = {}
                                    req_metrics = {"wall_time_s": round(wall_s, 6)}
                                    req_status = "error"
                                    req_error = str(exc)[:1000]
                                    req_error_traceback = traceback.format_exc(limit=20)
                                else:
                                    req_error_traceback = None
                                finally:
                                    sampler.stop()

                                peak_mb = sampler.peak_mb()
                                observed_request_mb = None
                                if peak_mb is not None and idle_loaded_mb is not None:
                                    observed_request_mb = max(0, peak_mb - idle_loaded_mb)

                                row = {
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "model_label": model_cfg.get("label", model_cfg["model"]),
                                    "model": model_cfg["model"],
                                    "tokenizer": model_cfg.get("tokenizer", model_cfg["model"]),
                                    "weight_quantization": model_cfg.get("weight_quantization"),
                                    "kv_config": kv_name,
                                    "kv_cache_dtype": KV_CONFIGS[kv_name]["kv_cache_dtype"],
                                    "vllm_effective_kv_storage_dtype": vllm_effective_kv_storage_dtype(kv_name),
                                    "context": context,
                                    "repeat": repeat,
                                    "question_id": q["id"],
                                    "category": q.get("category"),
                                    "passed": passed,
                                    "status": req_status,
                                    "error": req_error,
                                    "error_traceback": req_error_traceback,
                                    "answer": answer,
                                    "check_details": check_details,
                                    "load_wall_time_s": round(load_wall_s, 6),
                                    "gpu_idle_before_load_mb": idle_mb_before_load,
                                    "gpu_idle_loaded_model_mb": idle_loaded_mb,
                                    "gpu_peak_during_request_mb": peak_mb,
                                    "gpu_observed_request_mb": observed_request_mb,
                                    "gpu_util_peak_pct": sampler.peak_gpu_util_pct(),
                                    "gpu_util_mean_pct": sampler.mean_gpu_util_pct(),
                                    "gpu_memory_util_peak_pct": sampler.peak_memory_util_pct(),
                                    "gpu_memory_util_mean_pct": sampler.mean_memory_util_pct(),
                                    "gpu_sampler_available": sampler.available,
                                    "metrics": req_metrics,
                                }
                                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                                completed.add(row_key)

                    except Exception as exc:
                        status = "load_error"
                        error = str(exc)[:2000]
                        row = {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "model_label": model_cfg.get("label", model_cfg["model"]),
                            "model": model_cfg["model"],
                            "weight_quantization": model_cfg.get("weight_quantization"),
                            "kv_config": kv_name,
                            "kv_cache_dtype": KV_CONFIGS[kv_name]["kv_cache_dtype"],
                            "vllm_effective_kv_storage_dtype": vllm_effective_kv_storage_dtype(kv_name),
                            "context": context,
                            "status": status,
                            "error": error,
                            "error_traceback": traceback.format_exc(limit=20),
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    finally:
                        shutdown_llm(llm)


def summarize(path: Path) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            key = (row["model_label"], row["context"], row["kv_config"])
            groups.setdefault(key, []).append(row)

    summary = []
    mem_by_model_context: dict[tuple[str, int], dict[str, float | None]] = {}
    for (model, context, kv_config), rows in sorted(groups.items()):
        wall = [r["metrics"].get("wall_time_s") for r in rows if r.get("metrics", {}).get("wall_time_s") is not None]
        ttft = [r["metrics"].get("ttft_ms") for r in rows if r.get("metrics", {}).get("ttft_ms") is not None]
        request_mem = [r.get("gpu_observed_request_mb") for r in rows if r.get("gpu_observed_request_mb") is not None]
        gpu_util = [r.get("gpu_util_mean_pct") for r in rows if r.get("gpu_util_mean_pct") is not None]
        memory_util = [r.get("gpu_memory_util_mean_pct") for r in rows if r.get("gpu_memory_util_mean_pct") is not None]
        passed = sum(1 for r in rows if r.get("passed"))
        mem_median = round(median(request_mem), 1) if request_mem else None
        mem_by_model_context.setdefault((model, context), {})[kv_config] = mem_median
        summary.append(
            {
                "model": model,
                "context": context,
                "kv_config": kv_config,
                "n": len(rows),
                "pass_rate": round(passed / len(rows) * 100, 2) if rows else None,
                "wall_time_s_median": round(median(wall), 4) if wall else None,
                "wall_time_s_mean": round(mean(wall), 4) if wall else None,
                "ttft_ms_median": round(median(ttft), 3) if ttft else None,
                "gpu_observed_request_mb_median": mem_median,
                "gpu_util_mean_pct_median": round(median(gpu_util), 2) if gpu_util else None,
                "gpu_memory_util_mean_pct_median": round(median(memory_util), 2) if memory_util else None,
            }
        )

    for item in summary:
        baseline = mem_by_model_context.get((item["model"], item["context"]), {}).get("fp16")
        current = item["gpu_observed_request_mb_median"]
        if baseline is None or current is None or item["kv_config"] == "fp16":
            item["gpu_observed_memory_saved_mb_vs_fp16"] = None
            item["gpu_observed_memory_saved_pct_vs_fp16"] = None
            continue
        saved_mb = baseline - current
        item["gpu_observed_memory_saved_mb_vs_fp16"] = round(saved_mb, 1)
        item["gpu_observed_memory_saved_pct_vs_fp16"] = round(saved_mb / baseline * 100, 2) if baseline > 0 else None

    return {"summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path, default=None)
    parser.add_argument("--model-labels", nargs="+", default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--question-ids", nargs="+", default=None)
    parser.add_argument("--contexts", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--kv-configs", nargs="+", choices=list(KV_CONFIGS), default=["fp16", "fp8", "turboquant_4bit", "turboquant_3bit"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/vllm_4model_experiment.jsonl"))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume-skip-existing", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        print(json.dumps(summarize(args.output), indent=2))
        return

    if args.attention_backend is None and shutil.which("nvcc") is None:
        args.attention_backend = "TRITON_ATTN"

    run_suite(args)
    print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
