"""Reusable quality check examples for consistency testing.

Fixed prompts and expected behaviors so every run produces comparable
results. Use these to verify TurboQuant quality hasn't regressed.

Usage:
    python3 -m benchmarks.quality_check_examples
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer
from turboquant.utils.metrics import QualityMetrics


# ──────────────────────────────────────────────────────────────────────
#  Fixed test vectors for reproducible quality checks
# ──────────────────────────────────────────────────────────────────────

QUALITY_EXAMPLES = {
    "standard_kv": {
        "description": "Standard KV cache vectors (normal distribution)",
        "seed": 42,
        "batch": 1,
        "seq_len": 512,
        "num_kv_heads": 8,
        "head_dim": 128,
        "thresholds": {
            "4bit_key_cos_min": 0.99,
            "4bit_val_cos_min": 0.99,
            "4bit_key_snr_min": 20.0,
        },
    },
    "outlier_kv": {
        "description": "KV cache with 10% outlier channels (10x magnitude)",
        "seed": 42,
        "batch": 1,
        "seq_len": 512,
        "num_kv_heads": 8,
        "head_dim": 128,
        "outlier_channels": 13,  # 10% of 128
        "outlier_scale": 10.0,
        "thresholds": {
            "4bit_key_cos_min": 0.99,
            "4bit_val_cos_min": 0.99,
        },
    },
    "long_context": {
        "description": "Long context KV cache (4096 tokens)",
        "seed": 42,
        "batch": 1,
        "seq_len": 4096,
        "num_kv_heads": 8,
        "head_dim": 128,
        "thresholds": {
            "4bit_key_cos_min": 0.99,
            "4bit_val_cos_min": 0.99,
        },
    },
    "gqa_small_kv": {
        "description": "GQA with 2 KV heads (like Qwen2.5-0.5B)",
        "seed": 42,
        "batch": 1,
        "seq_len": 256,
        "num_kv_heads": 2,
        "head_dim": 64,
        "thresholds": {
            "4bit_key_cos_min": 0.99,
            "4bit_val_cos_min": 0.99,
        },
    },
    "bonsai_like": {
        "description": "Bonsai-8B architecture (36 layers, 8 KV heads, dim 128)",
        "seed": 42,
        "batch": 1,
        "seq_len": 512,
        "num_kv_heads": 8,
        "head_dim": 128,
        "outlier_channels": 13,
        "outlier_scale": 8.0,
        "thresholds": {
            "4bit_key_cos_min": 0.99,
            "4bit_val_cos_min": 0.99,
        },
    },
}

# Fixed prompts for generation consistency checks
GENERATION_PROMPTS = [
    {
        "id": "math_pythagorean",
        "prompt": "The Pythagorean theorem states that",
        "expected_contains": ["right", "triangle", "square"],
        "category": "math",
    },
    {
        "id": "factual_paris",
        "prompt": "The capital of France is Paris, which is known for",
        "expected_contains": ["tower", "Eiffel", "architecture", "museum"],
        "category": "factual",
    },
    {
        "id": "science_gravity",
        "prompt": "Newton's law of universal gravitation describes",
        "expected_contains": ["force", "mass", "attract"],
        "category": "science",
    },
    {
        "id": "code_function",
        "prompt": "In Python, to define a function you use the keyword",
        "expected_contains": ["def"],
        "category": "code",
    },
    {
        "id": "reasoning_math",
        "prompt": "If a train travels at 60 km/h for 2 hours, it covers",
        "expected_contains": ["120", "km", "kilometers"],
        "category": "reasoning",
    },
]


def generate_test_kv(example: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate reproducible test KV tensors from an example spec."""
    torch.manual_seed(example["seed"])
    keys = torch.randn(
        example["batch"], example["seq_len"],
        example["num_kv_heads"], example["head_dim"]
    )
    values = torch.randn(
        example["batch"], example["seq_len"],
        example["num_kv_heads"], example["head_dim"]
    )

    # Add outliers if specified
    if "outlier_channels" in example:
        n = example["outlier_channels"]
        s = example["outlier_scale"]
        keys[:, :, :, :n] *= s

    return keys, values


def run_quality_check(preset: str = "turbo_4bit") -> dict:
    """Run all quality check examples and return pass/fail for each."""
    print(f"{'='*60}")
    print(f"  Quality Check: {preset}")
    print(f"{'='*60}")

    results = {}
    all_pass = True

    for name, example in QUALITY_EXAMPLES.items():
        keys, values = generate_test_kv(example)

        config_fn = getattr(TurboQuantConfig, preset)
        config = config_fn(
            num_heads=example["num_kv_heads"] * 4,  # Assume GQA ratio of 4
            num_kv_heads=example["num_kv_heads"],
            head_dim=example["head_dim"],
        )
        quantizer = TurboQuantizer(config)

        # Encode/decode
        q_k, mk = quantizer.encode_keys(keys)
        q_v, mv = quantizer.encode_values(values)
        rk = quantizer.decode_keys(q_k, mk)
        rv = quantizer.decode_values(q_v, mv)

        # Metrics
        k_vec = QualityMetrics.vector_metrics(keys, rk)
        v_vec = QualityMetrics.vector_metrics(values, rv)
        k_elem = QualityMetrics.element_metrics(keys, rk)
        v_elem = QualityMetrics.element_metrics(values, rv)

        # Check thresholds
        passed = True
        checks = {}
        t = example["thresholds"]
        if "4bit_key_cos_min" in t:
            ok = k_vec["cosine_similarity_mean"] >= t["4bit_key_cos_min"]
            checks["key_cos"] = {"value": k_vec["cosine_similarity_mean"], "threshold": t["4bit_key_cos_min"], "pass": ok}
            passed = passed and ok
        if "4bit_val_cos_min" in t:
            ok = v_vec["cosine_similarity_mean"] >= t["4bit_val_cos_min"]
            checks["val_cos"] = {"value": v_vec["cosine_similarity_mean"], "threshold": t["4bit_val_cos_min"], "pass": ok}
            passed = passed and ok
        if "4bit_key_snr_min" in t:
            ok = k_elem["snr_db"] >= t["4bit_key_snr_min"]
            checks["key_snr"] = {"value": k_elem["snr_db"], "threshold": t["4bit_key_snr_min"], "pass": ok}
            passed = passed and ok

        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed

        print(f"\n  [{status}] {name}: {example['description']}")
        for check_name, check_data in checks.items():
            mark = "+" if check_data["pass"] else "X"
            print(f"    [{mark}] {check_name}: {check_data['value']:.4f} (threshold: {check_data['threshold']})")

        results[name] = {
            "description": example["description"],
            "passed": passed,
            "key_cosine_sim": k_vec["cosine_similarity_mean"],
            "value_cosine_sim": v_vec["cosine_similarity_mean"],
            "key_snr_db": k_elem["snr_db"],
            "value_snr_db": v_elem["snr_db"],
            "checks": checks,
        }

    print(f"\n{'='*60}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  {sum(1 for r in results.values() if r['passed'])}/{len(results)} examples passed")
    print(f"{'='*60}")

    return {"preset": preset, "all_pass": all_pass, "examples": results}


def main():
    print("TurboQuant Quality Check Examples")
    print("Reproducible, reusable consistency tests\n")

    all_results = {}
    for preset in ["turbo_4bit", "turbo_3bit", "kivi_2bit"]:
        r = run_quality_check(preset)
        all_results[preset] = r
        print()

    # Save
    with open("quality_check_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to quality_check_results.json")

    # Print prompt examples
    print(f"\n{'='*60}")
    print("  Saved Generation Prompts (for model testing)")
    print(f"{'='*60}")
    for p in GENERATION_PROMPTS:
        print(f"  [{p['id']}] ({p['category']})")
        print(f"    Prompt: \"{p['prompt']}\"")
        print(f"    Expected: {p['expected_contains']}")


if __name__ == "__main__":
    main()
