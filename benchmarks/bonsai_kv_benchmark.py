#!/usr/bin/env python3
"""
Bonsai-8B KV Cache Quantization Benchmark
==========================================
Runs Bonsai-8B (1-bit weights) through PrismML's llama-cli with
different KV cache quantization levels: f16, q8_0, q4_0

Uses the same 5 QA questions from the Bonsai benchmark suite to
measure quality impact of KV cache compression.

This is the real end-to-end test: 1-bit model weights + quantized KV cache.
"""

import subprocess
import json
import time
import os
import sys
import re

# ── Paths ──
BONSAI_DIR = "/Users/pradip/Desktop/Learning/Claude/PrismML/Bonsai-demo"
LLAMA_CLI = os.path.join(BONSAI_DIR, "bin", "mac", "llama-cli")
LLAMA_BENCH = os.path.join(BONSAI_DIR, "bin", "mac", "llama-bench")
MODEL_PATH = os.path.join(BONSAI_DIR, "models", "Bonsai-8B.gguf")
DYLD_PATH = os.path.join(BONSAI_DIR, "bin", "mac")

# ── QA Questions (from Bonsai qa_benchmark_v2.py) ──
QA_QUESTIONS = [
    {
        "id": 1,
        "cat": "Context QA",
        "question": 'Based on: "Meridian Tech reported Q3 revenue of $4.2B, up 18% YoY. Cloud division grew 34% to $1.8B. Operating margins improved to 22.5% from 19.1%."\n\nWhat were the operating margins in Q3 last year?',
        "accept_fn": lambda a: "19.1" in a,
    },
    {
        "id": 2,
        "cat": "Reasoning",
        "question": "A DNS outage at 9:00 AM caused the website to go down at 9:02 AM. Engineering switched to backup DNS at 9:45 AM and the site was restored at 9:47 AM. What was the total downtime?",
        "accept_fn": lambda a: "45" in a and ("min" in a.lower() or "minute" in a.lower()),
    },
    {
        "id": 3,
        "cat": "Instruction",
        "question": 'Convert to JSON with keys "name", "role", "city": "Alice Chen is a senior engineer based in Seattle." Return ONLY JSON.',
        "accept_fn": lambda a: '"name"' in a and '"Alice' in a and '"Seattle' in a,
    },
    {
        "id": 4,
        "cat": "Finance",
        "question": "A company has revenue of $50M, COGS of $30M, and operating expenses of $12M. What is the gross margin percentage and operating income?",
        "accept_fn": lambda a: ("40%" in a or "40 %" in a) and ("8" in a),
    },
    {
        "id": 5,
        "cat": "Code",
        "question": 'What does this print?\n```python\nx = [10, 20, 30, 40, 50]\nprint(x[1:3])\nprint(sum(x))\n```',
        "accept_fn": lambda a: "[20, 30]" in a and "150" in a,
    },
]

# ── KV Cache Configurations ──
KV_CONFIGS = [
    {"name": "f16",  "ctk": "f16",  "ctv": "f16",  "desc": "FP16 (baseline)"},
    {"name": "q8_0", "ctk": "q8_0", "ctv": "q8_0", "desc": "Q8_0 (8-bit uniform)"},
    {"name": "q4_0", "ctk": "q4_0", "ctv": "q4_0", "desc": "Q4_0 (4-bit uniform)"},
]

CONTEXT_SIZES = [1024, 4096, 16384, 32768]


def run_llama_qa(model_path, question, ctk="f16", ctv="f16", ctx=4096, max_tokens=200):
    """Run a QA question through llama-cli with specified KV cache type."""
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-ngl", "999",
        "-fa", "1",
        "-c", str(ctx),
        "-ctk", ctk,
        "-ctv", ctv,
        "-n", str(max_tokens),
        "-p", prompt,
        "--temp", "0",
        "--single-turn",
        "--no-display-prompt",
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = DYLD_PATH

    try:
        start = time.perf_counter()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        elapsed = time.perf_counter() - start

        full_output = result.stdout + "\n" + result.stderr

        # Clean answer: strip ANSI codes, loading noise
        output = result.stdout
        output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        output = re.sub(r'\[ Prompt:.*?]', '', output)
        for noise in ["Loading model...", "Exiting...", "llama_memory", "ggml_metal",
                       "build      :", "model      :", "modalities :",
                       "available commands:", "/exit", "/regen", "/clear", "/read"]:
            output = output.replace(noise, "")
        # Remove block art banner
        output = re.sub(r'[▄▀█░▓▒]+.*', '', output)
        output = re.sub(r'\|\\-/\|', '', output)
        output = re.sub(r'[\|\\\-/]{1,4}\s*', '', output)
        output = output.strip()

        # Parse throughput
        pp_tps = 0.0
        gen_tps = 0.0
        m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", full_output)
        if m:
            pp_tps = float(m.group(1))
            gen_tps = float(m.group(2))

        return {
            "answer": output[:500],
            "wall_s": round(elapsed, 2),
            "pp_tps": pp_tps,
            "gen_tps": gen_tps,
        }
    except subprocess.TimeoutExpired:
        return {"answer": "[TIMEOUT]", "wall_s": 60, "pp_tps": 0, "gen_tps": 0}
    except Exception as e:
        return {"answer": f"[ERROR: {e}]", "wall_s": 0, "pp_tps": 0, "gen_tps": 0}


def run_memory_bench(model_path, prompt_tokens, ctk="f16", ctv="f16"):
    """Run llama-bench to get memory usage at specific context."""
    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-ngl", "999",
        "-fa", "1",
        "-ctk", ctk,
        "-ctv", ctv,
        "-p", str(prompt_tokens),
        "-n", "1",
        "-r", "1",
        "-o", "json",
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = DYLD_PATH

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        full = result.stdout + "\n" + result.stderr

        # Parse JSON output
        for line in full.split("\n"):
            line = line.strip()
            if line.startswith("["):
                try:
                    data = json.loads(line)
                    if data:
                        return data[0]
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"    bench error: {e}")

    return None


def main():
    print("=" * 70)
    print("  Bonsai-8B KV Cache Quantization Benchmark")
    print("  1-bit weights + {f16, q8_0, q4_0} KV cache")
    print("=" * 70)

    if not os.path.exists(LLAMA_CLI):
        print(f"  ERROR: llama-cli not found at {LLAMA_CLI}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"  Model: Bonsai-8B (1-bit Q1_0_g128, {os.path.getsize(MODEL_PATH)/(1024**2):.0f} MB)")
    print(f"  Binary: {LLAMA_CLI}")

    all_results = {
        "benchmark": "Bonsai-8B + KV Cache Quantization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "Bonsai-8B",
        "model_size_mb": os.path.getsize(MODEL_PATH) / (1024**2),
    }

    # ── Test 1: QA Quality across KV configs ──
    print(f"\n{'='*70}")
    print(f"  Test 1: QA Quality with Different KV Cache Quantization")
    print(f"{'='*70}")

    qa_results = {}
    for kv in KV_CONFIGS:
        print(f"\n  --- KV Cache: {kv['name']} ({kv['desc']}) ---")
        results = []
        passed = 0

        for q in QA_QUESTIONS:
            r = run_llama_qa(MODEL_PATH, q["question"], ctk=kv["ctk"], ctv=kv["ctv"])
            is_pass = q["accept_fn"](r["answer"])
            if is_pass:
                passed += 1
            results.append({
                "id": q["id"],
                "cat": q["cat"],
                "passed": is_pass,
                "answer": r["answer"][:200],
                "wall_s": r["wall_s"],
                "gen_tps": r["gen_tps"],
            })
            mark = "PASS" if is_pass else "FAIL"
            print(f"    Q{q['id']} [{q['cat']:<12}] [{mark}] {r['wall_s']:.1f}s  {r['answer'][:80]}...")

        print(f"    Score: {passed}/{len(QA_QUESTIONS)}")
        qa_results[kv["name"]] = {"passed": passed, "total": len(QA_QUESTIONS), "results": results}

    all_results["qa"] = qa_results

    # ── Test 2: Memory at different context sizes ──
    print(f"\n{'='*70}")
    print(f"  Test 2: Memory Usage at Different Context Sizes")
    print(f"{'='*70}")

    memory_results = {}
    print(f"\n  {'KV Type':<8} {'Context':>8} {'Total MiB':>10} {'pp tok/s':>10}")
    print(f"  {'-'*40}")

    for kv in KV_CONFIGS:
        mem_data = []
        for ctx in CONTEXT_SIZES:
            bench = run_memory_bench(MODEL_PATH, ctx, ctk=kv["ctk"], ctv=kv["ctv"])
            if bench:
                total_mib = bench.get("mem_total_mib", 0)
                pp_tps = bench.get("pp_tps", 0)
                print(f"  {kv['name']:<8} {ctx:>8} {total_mib:>10.0f} {pp_tps:>10.1f}")
                mem_data.append({"ctx": ctx, "total_mib": total_mib, "pp_tps": pp_tps})
            else:
                print(f"  {kv['name']:<8} {ctx:>8}      --         --")
                mem_data.append({"ctx": ctx, "total_mib": 0, "pp_tps": 0})

        memory_results[kv["name"]] = mem_data

    all_results["memory"] = memory_results

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'KV Config':<12} {'QA Score':>10} {'Quality':>10}")
    print(f"  {'-'*35}")
    for kv in KV_CONFIGS:
        qr = qa_results[kv["name"]]
        pct = qr["passed"] / qr["total"] * 100
        quality = "SAME" if qr["passed"] == qa_results["f16"]["passed"] else (
            "BETTER" if qr["passed"] > qa_results["f16"]["passed"] else "DEGRADED"
        )
        print(f"  {kv['desc']:<25} {qr['passed']}/{qr['total']:>5} {quality:>10}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "bonsai_kv_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
