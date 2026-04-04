#!/usr/bin/env python3
"""Smoke test: Bonsai 1-bit + FP16 vs Q4_0 KV cache on production QA questions."""

import subprocess, os, re, time, sys

BONSAI_DIR = "/Users/pradip/Desktop/Learning/Claude/PrismML/Bonsai-demo"
LLAMA_CLI = os.path.join(BONSAI_DIR, "bin", "mac", "llama-cli")
DYLD = os.path.join(BONSAI_DIR, "bin", "mac")

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Bonsai-1.7B"
MODEL = os.path.join(BONSAI_DIR, "models", f"{MODEL_NAME}.gguf")


def run(q, ctk="f16", max_tok=400):
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    cmd = [
        LLAMA_CLI, "-m", MODEL, "-ngl", "999", "-fa", "1", "-c", "4096",
        "-ctk", ctk, "-ctv", ctk, "-n", str(max_tok), "-p", prompt,
        "--temp", "0", "--single-turn", "--no-display-prompt",
    ]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = DYLD
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    full = r.stdout + r.stderr
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", full)
    tps = float(m.group(2)) if m else 0
    # Clean
    ans = r.stdout
    for noise in ["Loading model...", "Exiting..."]:
        ans = ans.replace(noise, "")
    ans = re.sub(r"[▄▀█░▓▒]+[^\n]*", "", ans)
    ans = re.sub(r"\x1b\[[0-9;]*m", "", ans)
    ans = re.sub(r"\[.*?\]", "", ans)
    lines = []
    for l in ans.split("\n"):
        s = l.strip()
        if not s:
            continue
        skip = any(x in s for x in [">", "<|", "build", "model ", "modalities",
                                     "available", "/exit", "/regen", "/clear", "/read"])
        if not skip:
            lines.append(s)
    return "\n".join(lines).strip(), tps


QA = [
    {
        "id": 1, "cat": "Finance",
        "q": 'A company has revenue of $50M, COGS of $30M, and operating expenses of $12M. What is the gross margin percentage and operating income?',
        "check": lambda a: ("40" in a) and ("$8" in a or "8M" in a or "8 million" in a.lower() or "8,000" in a),
    },
    {
        "id": 2, "cat": "Reasoning",
        "q": "A DNS outage at 9:00 AM caused the website to go down at 9:02 AM. Engineering switched to backup DNS at 9:45 AM and the site was restored at 9:47 AM. What was the total downtime?",
        "check": lambda a: "45" in a,
    },
    {
        "id": 3, "cat": "Context QA",
        "q": 'Based on: "Meridian Tech reported Q3 revenue of $4.2B, up 18% YoY. Cloud division grew 34% to $1.8B. Operating margins improved to 22.5% from 19.1%." What were the operating margins in Q3 last year?',
        "check": lambda a: "19.1" in a,
    },
    {
        "id": 4, "cat": "Code",
        "q": "In Python, x = [10, 20, 30, 40, 50]. What is x[1:3] and what is sum(x)? Give the exact values.",
        "check": lambda a: "20" in a and "30" in a and "150" in a,
    },
    {
        "id": 5, "cat": "Instruction",
        "q": "Convert this to JSON with keys name, role, city: Alice Chen is a senior engineer based in Seattle. Return only the JSON object.",
        "check": lambda a: "Alice" in a and "Seattle" in a and ("engineer" in a.lower() or "senior" in a.lower()),
    },
]


def main():
    kv_quant = sys.argv[2] if len(sys.argv) > 2 else "q4_0"

    print(f"{MODEL_NAME}: Production QA -- FP16 vs {kv_quant} KV Cache")
    print("=" * 70)

    score_f16 = 0
    score_q4 = 0

    for q in QA:
        a16, t16 = run(q["q"], "f16")
        a4, t4 = run(q["q"], kv_quant)
        p16 = q["check"](a16)
        p4 = q["check"](a4)
        if p16:
            score_f16 += 1
        if p4:
            score_q4 += 1

        tag16 = "PASS" if p16 else "FAIL"
        tag4 = "PASS" if p4 else "FAIL"

        print(f"\n--- Q{q['id']} [{q['cat']}] ---")
        print(f"  FP16       [{tag16}] ({t16:.0f} tok/s):")
        print(f"    {a16[:350]}")
        print(f"  {kv_quant:<8}   [{tag4}] ({t4:.0f} tok/s):")
        print(f"    {a4[:350]}")

        if a16[:80] == a4[:80]:
            print("  Quality: IDENTICAL")
        elif p16 == p4:
            print("  Quality: SAME CORRECTNESS")
        else:
            print("  Quality: DIVERGED")

    print(f"\n{'='*70}")
    print(f"SCORE: FP16 = {score_f16}/5, {kv_quant} = {score_q4}/5")
    if score_f16 == score_q4:
        print(f"RESULT: {kv_quant} KV cache matches FP16 -- NO QUALITY LOSS")
    elif score_q4 >= score_f16 - 1:
        print(f"RESULT: {kv_quant} KV within 1 of FP16 -- MINIMAL DEGRADATION")
    else:
        print(f"RESULT: {kv_quant} degraded by {score_f16 - score_q4} questions")


if __name__ == "__main__":
    main()
