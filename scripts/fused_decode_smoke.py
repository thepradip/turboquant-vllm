#!/usr/bin/env python3
"""Focused vLLM TurboQuant fused-decode smoke runner."""

from __future__ import annotations

import argparse
import gc
import json
import time

import torch
from vllm import LLM, SamplingParams

import turboquant.vllm_plugin as tq_plugin


MODELS = {
    "qwen3.5-4b": {
        "model": "/mnt/hf-models/qwen3.5-4b-awq-4bit",
        "tokenizer": "/mnt/hf-models/qwen3.5-4b-awq-4bit",
        "quantization": "compressed-tensors",
    },
    "qwen3.5-9b": {
        "model": "/mnt/hf-models/qwen3.5-9b-awq-4bit",
        "tokenizer": "/mnt/hf-models/qwen3.5-9b-awq-4bit",
        "quantization": "compressed-tensors",
    },
    "gemma-4-e4b-it": {
        "model": "/mnt/hf-models/gemma-4-e4b-it-gptq-4bit",
        "tokenizer": "/mnt/hf-models/gemma-4-e4b-it-gptq-4bit",
        "quantization": "gptq",
    },
}

FILLER = (
    " Context filler. Ignore this text unless the question asks about it."
    " It is only here to occupy prompt tokens for a TurboQuant decode smoke test."
)
QUESTION = "What is 15 multiplied by 23? Give only the number."


def shutdown_llm(llm: LLM) -> None:
    try:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown(timeout=30)
    except Exception as exc:  # pragma: no cover - cleanup best effort
        print(f"Warning: shutdown failed: {exc}", flush=True)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_prompt(tokenizer: object, target_context: int, max_tokens: int) -> str:
    prompt = QUESTION
    try:
        while len(tokenizer.encode(prompt)) < target_context - max_tokens - 16:
            prompt = FILLER + prompt
    except Exception:
        pass
    return prompt


def run_one(label: str, context: int, max_tokens: int) -> dict[str, object]:
    model_cfg = MODELS[label]
    row: dict[str, object] = {
        "label": label,
        "context": context,
        "kv_cache_dtype": "turboquant_4bit",
        "status": "error",
    }
    llm = None
    try:
        llm = LLM(
            model=model_cfg["model"],
            tokenizer=model_cfg["tokenizer"],
            quantization=model_cfg["quantization"],
            kv_cache_dtype="turboquant_4bit",
            trust_remote_code=True,
            attention_backend="TRITON_ATTN",
            max_model_len=context,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            disable_log_stats=True,
        )
        tokenizer = llm.get_tokenizer()
        prompt = build_prompt(tokenizer, context, max_tokens)
        sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=1234)
        start = time.time()
        outputs = llm.generate([prompt], sampling)
        wall = time.time() - start
        first = outputs[0]
        answer = first.outputs[0].text if first.outputs else ""
        row.update(
            {
                "status": "ok",
                "wall_time_s": round(wall, 3),
                "answer": answer.strip(),
                "prompt_tokens": len(getattr(first, "prompt_token_ids", []) or []),
                "gen_tokens": len(getattr(first.outputs[0], "token_ids", []) or []),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if llm is not None:
            shutdown_llm(llm)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=["qwen3.5-4b", "qwen3.5-9b", "gemma-4-e4b-it"],
    )
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    tq_plugin.register()

    results = []
    for label in args.models:
        row = run_one(label, args.context, args.max_tokens)
        results.append(row)
        print("SMOKE_RESULT", json.dumps(row), flush=True)

    print("SMOKE_SUMMARY", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
