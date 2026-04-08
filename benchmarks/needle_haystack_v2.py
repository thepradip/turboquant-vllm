#!/usr/bin/env python3
"""
Robust Needle-in-a-Haystack Benchmark v2

Improvements over v1:
- Real text from HuggingFace (no repetition)
- 3 difficulty levels per context (easy, medium, hard)
- Needles blend with surrounding text style
- Distractor facts for hard questions
- 6 context lengths: 1K, 4K, 8K, 16K, 32K, 60K
- Programmatic verification with specific detail checks

Text sources:
- 1K, 4K: CNN/DailyMail news articles (unique per test)
- 8K, 16K: ArXiv scientific papers
- 32K, 60K: PG19 public domain books

Author: Pradip Tivhale, April 2026
Hardware: Apple M2 Pro, 16 GB
"""

import json
import time
import subprocess
import os
import sys
from dataclasses import dataclass, asdict

# ============================================================================
# REAL TEXT SOURCES (fetched from HuggingFace, no repetition)
# ============================================================================

CONTEXT_CONFIGS = {
    "1K": {
        "target_tokens": 1024,
        "source": "cnn_dailymail",
        "description": "Single news article (~1K tokens)",
    },
    "4K": {
        "target_tokens": 4096,
        "source": "cnn_dailymail",
        "description": "4 concatenated news articles (~4K tokens)",
    },
    "8K": {
        "target_tokens": 8192,
        "source": "arxiv",
        "description": "Scientific paper from ArXiv (~8K tokens)",
    },
    "16K": {
        "target_tokens": 16384,
        "source": "arxiv",
        "description": "Two ArXiv papers concatenated (~16K tokens)",
    },
    "32K": {
        "target_tokens": 32768,
        "source": "pg19",
        "description": "Chapter from a public domain book (~32K tokens)",
    },
    "60K": {
        "target_tokens": 60000,
        "source": "pg19",
        "description": "Large section from a public domain book (~60K tokens)",
    },
}

# ============================================================================
# NEEDLES: 3 per context length (easy, medium, hard), 18 total
# Each needle is a SINGLE SENTENCE that blends with surrounding text style.
# ============================================================================

NEEDLES = {
    # --- 1K context (news style) ---
    "1K": [
        {
            "id": "1K-easy",
            "difficulty": "easy",
            "needle": (
                "According to internal documents obtained by Reuters, "
                "the company allocated exactly $47.3 million to Project "
                "Nightingale, which was approved on March 14, 2023."
            ),
            "question": (
                "How much money was allocated to Project Nightingale "
                "and when was it approved?"
            ),
            "check": lambda a: "47.3" in a and ("march" in a.lower() or "2023" in a),
            "position": 0.5,
        },
        {
            "id": "1K-medium",
            "difficulty": "medium",
            "needle": (
                "The investigation revealed that 2,847 customer accounts "
                "were affected between January 8 and January 12, resulting "
                "in unauthorized transfers totaling $1.92 million."
            ),
            "question": (
                "How many customer accounts were affected and what was "
                "the total amount of unauthorized transfers?"
            ),
            "check": lambda a: "2,847" in a or "2847" in a,
            "position": 0.3,
        },
        {
            "id": "1K-hard",
            "difficulty": "hard",
            "needle": (
                "Chief Technology Officer Maria Santos confirmed that "
                "the backup recovery time was 4 hours and 22 minutes, "
                "exceeding the 2-hour SLA by 142 minutes."
            ),
            "distractor": (
                "The company spokesperson noted that recovery procedures "
                "were completed within the standard 2-hour window as "
                "outlined in their service agreement."
            ),
            "question": (
                "What was the actual backup recovery time reported by "
                "the CTO, and by how much did it exceed the SLA?"
            ),
            "check": lambda a: (
                ("4 hour" in a.lower() or "4:22" in a or "22 minute" in a.lower())
                and ("142" in a or "santos" in a.lower())
            ),
            "position": 0.7,
        },
    ],

    # --- 4K context (news style) ---
    "4K": [
        {
            "id": "4K-easy",
            "difficulty": "easy",
            "needle": (
                "Federal investigators confirmed that the shipment, "
                "weighing exactly 3,215 kilograms, departed from Port "
                "of Rotterdam on vessel MV Castellano on November 3, 2024."
            ),
            "question": (
                "What was the exact weight of the shipment and which "
                "vessel carried it from Rotterdam?"
            ),
            "check": lambda a: ("3,215" in a or "3215" in a) and "castellano" in a.lower(),
            "position": 0.5,
        },
        {
            "id": "4K-medium",
            "difficulty": "medium",
            "needle": (
                "Dr. Kenji Watanabe, lead researcher at the Osaka "
                "Institute, published findings showing that the compound "
                "reduced inflammation markers by 73.6% in the Phase IIb "
                "trial involving 1,284 participants."
            ),
            "question": (
                "What percentage did the compound reduce inflammation "
                "markers by, and how many participants were in the trial?"
            ),
            "check": lambda a: "73.6" in a and ("1,284" in a or "1284" in a),
            "position": 0.25,
        },
        {
            "id": "4K-hard",
            "difficulty": "hard",
            "needle": (
                "The internal audit found that Branch 7 in Phoenix "
                "processed 14,891 transactions on December 19 alone, "
                "which was 3.4 times the daily average and triggered "
                "compliance alert CA-2024-0892."
            ),
            "distractor": (
                "Branch operations across the Southwest region reported "
                "normal transaction volumes throughout the holiday "
                "period, with no unusual activity flagged by automated "
                "monitoring systems."
            ),
            "question": (
                "Which branch processed an unusually high number of "
                "transactions on December 19, how many transactions "
                "were there, and what compliance alert was triggered?"
            ),
            "check": lambda a: (
                ("14,891" in a or "14891" in a)
                and ("phoenix" in a.lower() or "branch 7" in a.lower())
                and "0892" in a
            ),
            "position": 0.6,
        },
    ],

    # --- 8K context (scientific style) ---
    "8K": [
        {
            "id": "8K-easy",
            "difficulty": "easy",
            "needle": (
                "The experimental results demonstrated that Model C "
                "achieved a BLEU score of 42.7 on the WMT-2024 "
                "benchmark, surpassing the previous state-of-the-art "
                "by 3.1 points."
            ),
            "question": (
                "What BLEU score did Model C achieve on WMT-2024 and "
                "by how many points did it beat the previous best?"
            ),
            "check": lambda a: "42.7" in a and "3.1" in a,
            "position": 0.5,
        },
        {
            "id": "8K-medium",
            "difficulty": "medium",
            "needle": (
                "Cross-validation on the held-out test set (n=5,372) "
                "yielded an F1 score of 0.891 for the transformer "
                "variant and 0.847 for the CNN baseline, with "
                "statistical significance at p < 0.001."
            ),
            "question": (
                "What F1 score did the transformer variant achieve on "
                "the held-out test set, and what was the sample size?"
            ),
            "check": lambda a: "0.891" in a or ".891" in a,
            "position": 0.35,
        },
        {
            "id": "8K-hard",
            "difficulty": "hard",
            "needle": (
                "Ablation study (Table 4) revealed that removing the "
                "attention pruning module decreased throughput from "
                "847 tokens/sec to 312 tokens/sec while paradoxically "
                "improving perplexity from 8.34 to 7.91 on the "
                "validation split."
            ),
            "distractor": (
                "As shown in our supplementary analysis, the pruning "
                "mechanism maintained consistent throughput across all "
                "evaluation benchmarks without measurable impact on "
                "perplexity scores."
            ),
            "question": (
                "According to the ablation study in Table 4, what "
                "happened to throughput and perplexity when the "
                "attention pruning module was removed?"
            ),
            "check": lambda a: (
                ("847" in a or "312" in a)
                and ("8.34" in a or "7.91" in a)
            ),
            "position": 0.75,
        },
    ],

    # --- 16K context (scientific style) ---
    "16K": [
        {
            "id": "16K-easy",
            "difficulty": "easy",
            "needle": (
                "The clinical trial registered as NCT-2024-88431 "
                "enrolled patients from 37 hospitals across 12 "
                "countries, with a primary endpoint of overall "
                "survival at 24 months."
            ),
            "question": (
                "What was the clinical trial registration number, "
                "how many hospitals participated, and across how "
                "many countries?"
            ),
            "check": lambda a: "88431" in a and "37" in a and "12" in a,
            "position": 0.5,
        },
        {
            "id": "16K-medium",
            "difficulty": "medium",
            "needle": (
                "Sensor array deployed at coordinates 34.0522N, "
                "118.2437W recorded a peak vibration amplitude of "
                "0.0847g at exactly 02:14:33 UTC on February 9, "
                "2024, which correlated with the seismic event "
                "catalogued as LA-2024-0041."
            ),
            "question": (
                "What was the peak vibration amplitude recorded by "
                "the sensor array, at what exact time, and what was "
                "the catalogue number of the seismic event?"
            ),
            "check": lambda a: "0.0847" in a and ("02:14" in a or "2:14" in a) and "0041" in a,
            "position": 0.2,
        },
        {
            "id": "16K-hard",
            "difficulty": "hard",
            "needle": (
                "The retrospective analysis of Patient Cohort D "
                "(n=892) found that the combination therapy of "
                "Drug X at 150mg and Drug Y at 75mg administered "
                "bi-weekly resulted in a 5-year remission rate of "
                "68.3%, compared to 41.7% for Drug X monotherapy."
            ),
            "distractor": (
                "Previous studies in similar patient populations "
                "demonstrated that standard monotherapy protocols "
                "achieved remission rates comparable to or exceeding "
                "those observed in combination therapy arms across "
                "multiple Phase III trials."
            ),
            "question": (
                "In Patient Cohort D, what was the 5-year remission "
                "rate for the combination therapy versus monotherapy, "
                "and what were the dosages?"
            ),
            "check": lambda a: "68.3" in a and "41.7" in a and ("150" in a or "75" in a),
            "position": 0.65,
        },
    ],

    # --- 32K context (book/narrative style) ---
    "32K": [
        {
            "id": "32K-easy",
            "difficulty": "easy",
            "needle": (
                "The old merchant whispered to his apprentice that "
                "the combination to the iron vault was 7-34-89-12 "
                "and that it must never be written down or shared "
                "with anyone outside the guild."
            ),
            "question": (
                "What was the combination to the iron vault that "
                "the merchant whispered to his apprentice?"
            ),
            "check": lambda a: "7" in a and "34" in a and "89" in a and "12" in a,
            "position": 0.5,
        },
        {
            "id": "32K-medium",
            "difficulty": "medium",
            "needle": (
                "Among the cargo manifests recovered from the wreck, "
                "one entry stood out: 847 bolts of Venetian silk, "
                "valued at 12,400 ducats, consigned to the House of "
                "Medici and bearing the seal of Captain Lorenzo Vettori."
            ),
            "question": (
                "How many bolts of Venetian silk were listed in the "
                "cargo manifest, what was their value, and who was "
                "the captain?"
            ),
            "check": lambda a: "847" in a and ("12,400" in a or "12400" in a) and "vettori" in a.lower(),
            "position": 0.4,
        },
        {
            "id": "32K-hard",
            "difficulty": "hard",
            "needle": (
                "The surveyor's report, dated October 17, 1847, "
                "recorded that the north boundary of the estate "
                "measured exactly 2,341 feet and the south boundary "
                "measured 2,187 feet, with the discrepancy attributed "
                "to the creek bed shifting 154 feet eastward since "
                "the original 1802 survey."
            ),
            "distractor": (
                "Local records confirmed that the estate boundaries "
                "had remained unchanged since the original survey, "
                "with both the north and south boundaries measuring "
                "within standard tolerances of the 1802 measurements."
            ),
            "question": (
                "According to the surveyor's 1847 report, what were "
                "the exact measurements of the north and south "
                "boundaries, and how far had the creek shifted?"
            ),
            "check": lambda a: "2,341" in a or "2341" in a,
            "position": 0.8,
        },
    ],

    # --- 60K context (book/narrative style) ---
    "60K": [
        {
            "id": "60K-easy",
            "difficulty": "easy",
            "needle": (
                "The letter, postmarked from Vienna on June 23, "
                "1891, contained a bank draft for exactly 4,750 "
                "Austrian florins payable to one Friedrich Engel "
                "of 14 Bergstrasse, Salzburg."
            ),
            "question": (
                "What amount was the bank draft for in the Vienna "
                "letter, and who was the payee?"
            ),
            "check": lambda a: "4,750" in a or "4750" in a,
            "position": 0.5,
        },
        {
            "id": "60K-medium",
            "difficulty": "medium",
            "needle": (
                "Hidden in the third drawer of the oak escritoire "
                "was a folded map showing that the mine entrance "
                "lay at precisely 47 degrees 12 minutes north, "
                "11 degrees 23 minutes east, marked with a red "
                "cross and the initials J.K.S."
            ),
            "question": (
                "What were the exact coordinates of the mine "
                "entrance shown on the hidden map, and whose "
                "initials were on it?"
            ),
            "check": lambda a: ("47" in a and "12" in a) and "j.k.s" in a.lower().replace(" ", ""),
            "position": 0.15,
        },
        {
            "id": "60K-hard",
            "difficulty": "hard",
            "needle": (
                "The inventory of the apothecary, taken after the "
                "fire of September 1847, listed 2,891 glass vials "
                "destroyed, 147 intact, and exactly 23 containing "
                "a mercury compound that Inspector Hoffman ordered "
                "sealed under case number V-1847-0334."
            ),
            "distractor": (
                "Fire brigade records indicated that the apothecary's "
                "stock was largely preserved through the efforts of "
                "the volunteer company, with only minor losses "
                "reported to the municipal authorities."
            ),
            "question": (
                "After the apothecary fire, how many glass vials "
                "were destroyed, how many were intact, and what "
                "case number did Inspector Hoffman assign?"
            ),
            "check": lambda a: ("2,891" in a or "2891" in a) and "147" in a and "0334" in a,
            "position": 0.85,
        },
    ],
}


def fetch_real_text(source: str, target_tokens: int, offset: int = 0) -> str:
    """Fetch real text from HuggingFace datasets. No repetition."""
    from datasets import load_dataset

    target_chars = target_tokens * 4  # ~4 chars per token

    if source == "cnn_dailymail":
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
        text_parts = []
        total_chars = 0
        for i, sample in enumerate(ds):
            if i < offset:
                continue
            text_parts.append(sample["article"])
            total_chars += len(sample["article"])
            if total_chars >= target_chars:
                break
        return "\n\n".join(text_parts)[:target_chars]

    elif source == "arxiv":
        ds = load_dataset("ccdv/arxiv-summarization", split="test", streaming=True)
        text_parts = []
        total_chars = 0
        for i, sample in enumerate(ds):
            if i < offset:
                continue
            text_parts.append(sample["article"])
            total_chars += len(sample["article"])
            if total_chars >= target_chars:
                break
        return "\n\n".join(text_parts)[:target_chars]

    elif source == "pg19":
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        for i, sample in enumerate(ds):
            if i < offset:
                continue
            if len(sample["text"]) >= target_chars:
                # Take a chunk from the middle (skip headers)
                start = len(sample["text"]) // 4
                return sample["text"][start:start + target_chars]
        return ""

    raise ValueError(f"Unknown source: {source}")


def insert_needle(text: str, needle: str, position: float, distractor: str = None) -> str:
    """Insert needle (and optional distractor) into text at specified position."""
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 3:
        paragraphs = text.split("\n")

    # Insert needle
    needle_idx = max(1, min(int(len(paragraphs) * position), len(paragraphs) - 1))
    paragraphs.insert(needle_idx, needle)

    # Insert distractor at a different position (before the needle)
    if distractor:
        dist_idx = max(1, int(len(paragraphs) * max(0, position - 0.3)))
        paragraphs.insert(dist_idx, distractor)

    return "\n\n".join(paragraphs)


def build_prompt(document: str, question: str) -> str:
    """Build the retrieval prompt."""
    return (
        "Read the following document carefully. Answer the question at the end "
        "using ONLY information found in the document. Be specific and include "
        "exact numbers, names, and dates from the document.\n\n"
        "--- DOCUMENT START ---\n"
        f"{document}\n"
        "--- DOCUMENT END ---\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def parse_llama_metrics(stderr: str) -> dict:
    """Parse all performance metrics from llama.cpp stderr output."""
    metrics = {
        "prompt_tps": 0.0,
        "gen_tps": 0.0,
        "prompt_tokens": 0,
        "gen_tokens": 0,
        "prompt_time_ms": 0.0,
        "gen_time_ms": 0.0,
        "kv_cache_mib": 0.0,
        "model_mib": 0.0,
        "total_mib": 0.0,
    }
    for line in stderr.split("\n"):
        ll = line.lower()
        # Prompt eval speed
        if "prompt eval" in ll and "token" in ll and "per second" in ll:
            parts = line.split()
            for i, p in enumerate(parts):
                if "token" in p.lower() and i > 0:
                    try: metrics["prompt_tps"] = float(parts[i-1])
                    except: pass
        # Prompt eval time
        if "prompt eval time" in ll:
            parts = line.split("=")
            if len(parts) >= 2:
                try:
                    time_part = parts[1].strip().split()[0]
                    metrics["prompt_time_ms"] = float(time_part)
                except: pass
            # Token count
            for p in line.split():
                if "token" in p.lower():
                    try:
                        idx = line.split().index(p)
                        metrics["prompt_tokens"] = int(line.split()[idx-1])
                    except: pass
        # Generation speed
        if "eval time" in ll and "prompt" not in ll and "token" in ll and "per second" in ll:
            parts = line.split()
            for i, p in enumerate(parts):
                if "token" in p.lower() and i > 0:
                    try: metrics["gen_tps"] = float(parts[i-1])
                    except: pass
        # Generation time
        if "eval time" in ll and "prompt" not in ll:
            parts = line.split("=")
            if len(parts) >= 2:
                try:
                    time_part = parts[1].strip().split()[0]
                    metrics["gen_time_ms"] = float(time_part)
                except: pass
        # KV cache size
        if "kv" in ll and ("buffer" in ll or "cache" in ll) and "mib" in ll:
            for p in line.split():
                try:
                    val = float(p)
                    if 1 < val < 100000:
                        metrics["kv_cache_mib"] = val
                except: pass
        # Model size
        if "model size" in ll or "model_size" in ll:
            for p in line.split():
                try:
                    val = float(p)
                    if val > 10:
                        metrics["model_mib"] = val
                except: pass
    # Also parse the chat-format speed line: [ Prompt: 216.0 t/s | Generation: 70.8 t/s ]
    for line in stderr.split("\n"):
        if "Prompt:" in line and "t/s" in line and "Generation:" in line:
            parts = line.replace("[", "").replace("]", "").split("|")
            for p in parts:
                p = p.strip()
                if p.startswith("Prompt:"):
                    try: metrics["prompt_tps"] = float(p.split(":")[1].strip().split()[0])
                    except: pass
                elif p.startswith("Generation:"):
                    try: metrics["gen_tps"] = float(p.split(":")[1].strip().split()[0])
                    except: pass

    # TTFT = time to first token = prefill latency
    if metrics["prompt_tps"] > 0 and metrics["prompt_tokens"] > 0:
        metrics["ttft_ms"] = round(metrics["prompt_tokens"] / metrics["prompt_tps"] * 1000, 1)
    else:
        metrics["ttft_ms"] = metrics["prompt_time_ms"]

    # Prefill latency = prompt eval time (same as TTFT)
    metrics["prefill_latency_ms"] = metrics["prompt_time_ms"] if metrics["prompt_time_ms"] > 0 else metrics["ttft_ms"]

    # Decode latency = generation eval time
    metrics["decode_latency_ms"] = metrics["gen_time_ms"]

    # TTLT = time to last token = prefill + decode
    metrics["ttlt_ms"] = round(metrics["prefill_latency_ms"] + metrics["decode_latency_ms"], 1)

    # End-to-end tokens/sec = total tokens / total time
    total_tokens = metrics["prompt_tokens"] + metrics.get("gen_tokens", 0)
    total_time_s = (metrics["prefill_latency_ms"] + metrics["decode_latency_ms"]) / 1000.0
    metrics["e2e_tps"] = round(total_tokens / total_time_s, 1) if total_time_s > 0 else 0

    return metrics


def run_inference(llama_cli, model, prompt, context_size, kv_config, timeout=600):
    """Run llama-cli inference and capture all metrics."""
    # Write prompt to temp file (avoids shell escaping issues)
    prompt_file = "/tmp/needle_prompt.txt"
    with open(prompt_file, "w") as pf:
        pf.write(prompt)

    cmd = [
        llama_cli, "-m", model,
        "-c", str(context_size),
        "-n", "200",
        "-st",           # single turn, exit after response
        "-f", prompt_file,
    ]
    if kv_config != "f16":
        cmd.extend(["-ctk", kv_config, "-ctv", kv_config])

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        wall_time = round(time.time() - start, 2)
        # Parse metrics from both stdout (chat format) and stderr
        full_output = result.stdout + "\n" + result.stderr
        metrics = parse_llama_metrics(full_output)
        # Extract answer: everything after the last ">" prompt line
        stdout_lines = result.stdout.split("\n")
        answer_lines = []
        found_prompt = False
        for line in stdout_lines:
            if found_prompt and line.strip() and not line.startswith("[") and not line.startswith("Exiting"):
                answer_lines.append(line.strip())
            if line.strip().startswith(">"):
                found_prompt = True
                answer_lines = []
        answer_text = " ".join(answer_lines)
        return {
            "output": answer_text if answer_text else result.stdout.strip(),
            "stderr_raw": result.stderr[-500:] if result.stderr else "",
            "wall_time": wall_time,
            "status": "OK",
            "prompt_tps": metrics["prompt_tps"],
            "gen_tps": metrics["gen_tps"],
            "ttft_ms": metrics["ttft_ms"],
            "ttlt_ms": metrics["ttlt_ms"],
            "prefill_latency_ms": metrics["prefill_latency_ms"],
            "decode_latency_ms": metrics["decode_latency_ms"],
            "kv_cache_mib": metrics["kv_cache_mib"],
            "prompt_tokens": metrics["prompt_tokens"],
            "e2e_tps": metrics["e2e_tps"],
            "kv_bytes_per_token": round(metrics["kv_cache_mib"] * 1024 * 1024 / max(metrics["prompt_tokens"], 1), 1) if metrics["kv_cache_mib"] > 0 else 0,
        }
    except subprocess.TimeoutExpired:
        z = {"output": "", "wall_time": timeout, "status": "TIMEOUT"}
        for k in ["prompt_tps","gen_tps","ttft_ms","ttlt_ms","prefill_latency_ms","decode_latency_ms","kv_cache_mib","prompt_tokens","e2e_tps","kv_bytes_per_token"]:
            z[k] = 0
        return z
    except Exception as e:
        z = {"output": "", "wall_time": 0, "status": f"ERROR: {e}"}
        for k in ["prompt_tps","gen_tps","ttft_ms","ttlt_ms","prefill_latency_ms","decode_latency_ms","kv_cache_mib","prompt_tokens","e2e_tps","kv_bytes_per_token"]:
            z[k] = 0
        return z


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robust Needle-in-a-Haystack v2")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--llama-cli", default="./bin/mac/llama-cli")
    parser.add_argument("--contexts", nargs="+", default=["1K", "4K", "8K", "16K", "32K", "60K"])
    parser.add_argument("--kv-configs", nargs="+", default=["f16", "q4_0", "q3_0"])
    parser.add_argument("--output", default="needle_v2_results.json")
    parser.add_argument("--dry-run", action="store_true", help="Show dataset info without running inference")
    args = parser.parse_args()

    print("=" * 70)
    print("Needle-in-a-Haystack Benchmark v2 (Robust)")
    print("Real text from HuggingFace | No repetition | 3 difficulties")
    print("=" * 70)

    results = {
        "benchmark": "Needle-in-a-Haystack v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "design": {
            "text_sources": {
                "1K-4K": "CNN/DailyMail (real news articles)",
                "8K-16K": "ArXiv (real scientific papers)",
                "32K-60K": "PG19 (real public domain books)",
            },
            "needles_per_context": "3 (easy, medium, hard)",
            "total_needles": len(args.contexts) * 3,
            "distractor_facts": "present in all hard questions",
            "verification": "programmatic lambda checks on specific details",
        },
        "tests": [],
        "scores": {},
    }

    # Fetch real text for each context
    print("\nFetching real text from HuggingFace...")
    text_cache = {}
    offset_counter = {"cnn_dailymail": 0, "arxiv": 0, "pg19": 0}

    for ctx_label in args.contexts:
        cfg = CONTEXT_CONFIGS[ctx_label]
        source = cfg["source"]
        print(f"  {ctx_label}: {cfg['description']} from {source}...", end=" ", flush=True)
        text = fetch_real_text(source, cfg["target_tokens"], offset=offset_counter[source])
        offset_counter[source] += 10  # Skip ahead for next fetch (no overlap)
        text_cache[ctx_label] = text
        actual_tokens = len(text) // 4
        print(f"got ~{actual_tokens} tokens ({len(text)} chars)")

    if args.dry_run:
        print("\n--- DRY RUN: Dataset Preview ---")
        for ctx_label in args.contexts:
            text = text_cache[ctx_label]
            needles = NEEDLES[ctx_label]
            print(f"\n{'='*60}")
            print(f"Context: {ctx_label} (~{len(text)//4} tokens)")
            print(f"Text preview: {text[:200]}...")
            print(f"Needles:")
            for n in needles:
                print(f"  [{n['difficulty']}] {n['needle'][:80]}...")
                print(f"    Q: {n['question'][:80]}...")
                if "distractor" in n:
                    print(f"    Distractor: {n['distractor'][:60]}...")
        return

    # Run tests
    total = len(args.kv_configs) * len(args.contexts) * 3
    current = 0

    for kv_config in args.kv_configs:
        scores_key = f"kv_{kv_config}"
        results["scores"][scores_key] = {}

        for ctx_label in args.contexts:
            # Skip FP16 at 60K (won't fit in 16GB)
            if kv_config == "f16" and ctx_label == "60K":
                print(f"  SKIP: {kv_config} @ {ctx_label} (FP16 at 60K won't fit in 16GB)")
                results["scores"][scores_key][ctx_label] = {
                    "found": 0, "total": 0, "pct": 0, "skipped": True,
                }
                continue

            cfg = CONTEXT_CONFIGS[ctx_label]
            ctx_found = 0
            ctx_total = 0
            needles = NEEDLES[ctx_label]
            base_text = text_cache[ctx_label]

            for needle_info in needles:
                current += 1
                nid = needle_info["id"]
                diff = needle_info["difficulty"]

                print(f"[{current}/{total}] {kv_config} | {ctx_label} | {diff} | {nid}", end=" ... ", flush=True)

                # Insert needle (and distractor if hard)
                doc = insert_needle(
                    base_text,
                    needle_info["needle"],
                    needle_info["position"],
                    needle_info.get("distractor"),
                )
                prompt = build_prompt(doc, needle_info["question"])

                # Run inference
                ctx_size = cfg["target_tokens"] + 1024  # headroom
                resp = run_inference(
                    args.llama_cli, args.model, prompt,
                    ctx_size, kv_config,
                )

                found = resp["status"] == "OK" and needle_info["check"](resp["output"])
                ctx_found += int(found)
                ctx_total += 1

                result_entry = {
                    "kv_config": kv_config,
                    "context": ctx_label,
                    "needle_id": nid,
                    "difficulty": diff,
                    "found": found,
                    "wall_time_s": resp["wall_time"],
                    "ttft_ms": resp["ttft_ms"],
                    "ttlt_ms": resp["ttlt_ms"],
                    "prefill_latency_ms": resp["prefill_latency_ms"],
                    "decode_latency_ms": resp["decode_latency_ms"],
                    "prompt_tps": resp["prompt_tps"],
                    "gen_tps": resp["gen_tps"],
                    "prompt_tokens": resp["prompt_tokens"],
                    "kv_cache_mib": resp["kv_cache_mib"],
                    "e2e_tps": resp["e2e_tps"],
                    "kv_bytes_per_token": resp["kv_bytes_per_token"],
                    "status": resp["status"],
                    "answer_preview": resp["output"][:300],
                }
                results["tests"].append(result_entry)
                status_str = "FOUND" if found else "MISSED"
                print(
                    f"{status_str} | wall={resp['wall_time']}s "
                    f"TTFT={resp['ttft_ms']:.0f}ms "
                    f"TTLT={resp['ttlt_ms']:.0f}ms "
                    f"prefill={resp['prefill_latency_ms']:.0f}ms "
                    f"decode={resp['decode_latency_ms']:.0f}ms "
                    f"PP={resp['prompt_tps']:.0f}tok/s "
                    f"Gen={resp['gen_tps']:.0f}tok/s"
                )

            results["scores"][scores_key][ctx_label] = {
                "found": ctx_found,
                "total": ctx_total,
                "pct": round(100 * ctx_found / ctx_total, 1),
            }
            print(f"  >> {kv_config} @ {ctx_label}: {ctx_found}/{ctx_total}\n")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Compute per-config aggregated metrics
    import statistics
    results["summary"] = {}
    for kv_config in args.kv_configs:
        config_tests = [t for t in results["tests"] if t["kv_config"] == kv_config]
        if not config_tests:
            continue
        ok_tests = [t for t in config_tests if t["status"] == "OK"]
        found_tests = [t for t in ok_tests if t["found"]]

        # Percentile latencies
        wall_times = sorted([t["wall_time_s"] for t in ok_tests]) if ok_tests else [0]
        def percentile(data, p):
            if not data: return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return round(data[f] + (k - f) * (data[c] - data[f]), 2)

        results["summary"][f"kv_{kv_config}"] = {
            "total_tests": len(config_tests),
            "total_found": len(found_tests),
            "accuracy_pct": round(100 * len(found_tests) / len(ok_tests), 1) if ok_tests else 0,
            "avg_wall_time_s": round(sum(t["wall_time_s"] for t in ok_tests) / len(ok_tests), 2) if ok_tests else 0,
            "p50_wall_s": percentile(wall_times, 50),
            "p95_wall_s": percentile(wall_times, 95),
            "p99_wall_s": percentile(wall_times, 99),
            "avg_prompt_tps": round(sum(t["prompt_tps"] for t in ok_tests) / len(ok_tests), 1) if ok_tests else 0,
            "avg_gen_tps": round(sum(t["gen_tps"] for t in ok_tests) / len(ok_tests), 1) if ok_tests else 0,
            "avg_e2e_tps": round(sum(t["e2e_tps"] for t in ok_tests) / len(ok_tests), 1) if ok_tests else 0,
            "avg_ttft_ms": round(sum(t["ttft_ms"] for t in ok_tests) / len(ok_tests), 1) if ok_tests else 0,
            "avg_ttlt_ms": round(sum(t["ttlt_ms"] for t in ok_tests) / len(ok_tests), 1) if ok_tests else 0,
            "avg_prefill_ms": round(sum(t["prefill_latency_ms"] for t in ok_tests) / len(ok_tests), 1) if ok_tests else 0,
            "avg_decode_ms": round(sum(t["decode_latency_ms"] for t in ok_tests) / len(ok_tests), 1) if ok_tests else 0,
            "by_difficulty": {},
            "by_context": {},
        }

        for diff in ["easy", "medium", "hard"]:
            diff_tests = [t for t in ok_tests if t["difficulty"] == diff]
            if diff_tests:
                results["summary"][f"kv_{kv_config}"]["by_difficulty"][diff] = {
                    "found": sum(1 for t in diff_tests if t["found"]),
                    "total": len(diff_tests),
                    "pct": round(100 * sum(1 for t in diff_tests if t["found"]) / len(diff_tests), 1),
                }

        for ctx_label in args.contexts:
            ctx_tests = [t for t in ok_tests if t["context"] == ctx_label]
            if ctx_tests:
                results["summary"][f"kv_{kv_config}"]["by_context"][ctx_label] = {
                    "found": sum(1 for t in ctx_tests if t["found"]),
                    "total": len(ctx_tests),
                    "avg_wall_s": round(sum(t["wall_time_s"] for t in ctx_tests) / len(ctx_tests), 2),
                    "avg_pp_tps": round(sum(t["prompt_tps"] for t in ctx_tests) / len(ctx_tests), 1),
                    "avg_gen_tps": round(sum(t["gen_tps"] for t in ctx_tests) / len(ctx_tests), 1),
                    "avg_ttft_ms": round(sum(t["ttft_ms"] for t in ctx_tests) / len(ctx_tests), 1),
                }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for config_key, summary in results["summary"].items():
        s = summary
        print(f"\n  {config_key}: {s['total_found']}/{s['total_tests']} ({s['accuracy_pct']}%)")
        print(f"    Latency:  avg={s['avg_wall_time_s']}s  p50={s['p50_wall_s']}s  p95={s['p95_wall_s']}s  p99={s['p99_wall_s']}s")
        print(f"    TTFT={s['avg_ttft_ms']}ms  TTLT={s['avg_ttlt_ms']}ms  prefill={s['avg_prefill_ms']}ms  decode={s['avg_decode_ms']}ms")
        print(f"    Throughput:  PP={s['avg_prompt_tps']}tok/s  Gen={s['avg_gen_tps']}tok/s  E2E={s['avg_e2e_tps']}tok/s")
        print(f"    By difficulty:")
        for diff, d in s["by_difficulty"].items():
            print(f"      {diff}: {d['found']}/{d['total']} ({d['pct']}%)")
        print(f"    By context:")
        for ctx, c in s["by_context"].items():
            print(f"      {ctx}: {c['found']}/{c['total']} | wall={c['avg_wall_s']}s PP={c['avg_pp_tps']}tok/s Gen={c['avg_gen_tps']}tok/s TTFT={c['avg_ttft_ms']}ms")


if __name__ == "__main__":
    main()
