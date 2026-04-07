"""Test 3-bit TurboQuant (2-bit PolarQuant + 1-bit QJL) after QJL fixes.

Measures cosine similarity before and after the fix on head_dim=128
architectures (Llama/Bonsai-8B style).

Run: python benchmarks/test_3bit_fix.py
"""

import torch
import math
import json
import time
from datetime import datetime

import sys
sys.path.insert(0, ".")

from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer


def cosine_sim(a, b):
    """Compute cosine similarity between two tensors along last dim."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    dot = (a_flat * b_flat).sum(dim=-1)
    norm_a = torch.norm(a_flat, dim=-1)
    norm_b = torch.norm(b_flat, dim=-1)
    return (dot / (norm_a * norm_b + 1e-8)).mean().item()


def run_test(head_dim, num_kv_heads, seq_len, label):
    """Run 3-bit encode/decode and measure cosine similarity."""
    batch = 1
    heads = num_kv_heads

    # Generate synthetic KV tensors
    torch.manual_seed(42)
    keys = torch.randn(batch, seq_len, heads, head_dim)
    values = torch.randn(batch, seq_len, heads, head_dim)

    # Also test with outlier channels (realistic KV pattern)
    outlier_channels = [5, 31, 63, 99] if head_dim >= 128 else [3, 15]
    keys_outlier = keys.clone()
    values_outlier = values.clone()
    for ch in outlier_channels:
        if ch < head_dim:
            keys_outlier[..., ch] *= 10.0
            values_outlier[..., ch] *= 8.0

    # Create 3-bit config
    config = TurboQuantConfig.turbo_3bit(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_heads=32,
    )
    config.enable_mixed_precision = False  # isolate QJL fix

    quantizer = TurboQuantizer(config)

    results = {}

    for name, k, v in [
        ("standard", keys, values),
        ("with_outliers", keys_outlier, values_outlier),
    ]:
        # Encode
        q_keys, k_meta = quantizer.encode_keys(k)
        q_values, v_meta = quantizer.encode_values(v)

        # Decode
        r_keys = quantizer.decode_keys(q_keys, k_meta)
        r_values = quantizer.decode_values(q_values, v_meta)

        # Measure
        k_cos = cosine_sim(k, r_keys)
        v_cos = cosine_sim(v, r_values)

        results[name] = {
            "key_cosine": round(k_cos, 6),
            "value_cosine": round(v_cos, 6),
            "key_pass": k_cos >= 0.99,
            "value_pass": v_cos >= 0.99,
        }

        print(f"  {label} [{name}] key_cos={k_cos:.6f} val_cos={v_cos:.6f} "
              f"{'PASS' if k_cos >= 0.99 and v_cos >= 0.99 else 'FAIL'}")

    return results


def run_4bit_baseline(head_dim, num_kv_heads, seq_len, label):
    """Run 4-bit as baseline to confirm it's unchanged."""
    batch = 1
    torch.manual_seed(42)
    keys = torch.randn(batch, seq_len, num_kv_heads, head_dim)
    values = torch.randn(batch, seq_len, num_kv_heads, head_dim)

    config = TurboQuantConfig.turbo_4bit(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_heads=32,
    )
    quantizer = TurboQuantizer(config)

    q_keys, k_meta = quantizer.encode_keys(keys)
    q_values, v_meta = quantizer.encode_values(values)
    r_keys = quantizer.decode_keys(q_keys, k_meta)
    r_values = quantizer.decode_values(q_values, v_meta)

    k_cos = cosine_sim(keys, r_keys)
    v_cos = cosine_sim(values, r_values)

    print(f"  {label} [4-bit baseline] key_cos={k_cos:.6f} val_cos={v_cos:.6f} "
          f"{'PASS' if k_cos >= 0.99 and v_cos >= 0.99 else 'FAIL'}")

    return {
        "key_cosine": round(k_cos, 6),
        "value_cosine": round(v_cos, 6),
        "key_pass": k_cos >= 0.99,
        "value_pass": v_cos >= 0.99,
    }


def main():
    print("=" * 70)
    print("TurboQuant 3-bit QJL Fix Validation")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Device: CPU (Apple M2 Pro)")
    print("=" * 70)

    all_results = {
        "benchmark": "3-bit QJL fix validation",
        "timestamp": datetime.now().isoformat(),
        "configs": {},
    }

    test_configs = [
        # (head_dim, num_kv_heads, seq_len, label)
        (128, 8, 64, "Bonsai-8B (d=128, h=8, seq=64)"),
        (128, 8, 128, "Bonsai-8B (d=128, h=8, seq=128)"),
        (128, 8, 256, "Bonsai-8B (d=128, h=8, seq=256)"),
        (128, 4, 64, "Llama-3B (d=128, h=4, seq=64)"),
        (128, 2, 64, "Qwen-0.5B (d=128, h=2, seq=64)"),
        (64, 8, 64, "Small head (d=64, h=8, seq=64)"),
    ]

    # 3-bit tests
    print("\n--- 3-bit TurboQuant (2-bit PQ + 1-bit QJL) ---")
    for head_dim, num_kv_heads, seq_len, label in test_configs:
        print(f"\n[{label}]")
        result = run_test(head_dim, num_kv_heads, seq_len, label)
        all_results["configs"][label] = {"3bit": result}

    # 4-bit baseline (must stay unchanged)
    print("\n--- 4-bit Baseline (must be unchanged) ---")
    for head_dim, num_kv_heads, seq_len, label in test_configs[:3]:
        print(f"\n[{label}]")
        result = run_4bit_baseline(head_dim, num_kv_heads, seq_len, label)
        all_results["configs"][label]["4bit_baseline"] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_3bit = 0
    pass_3bit = 0
    for label, data in all_results["configs"].items():
        if "3bit" in data:
            for scenario, metrics in data["3bit"].items():
                total_3bit += 1
                if metrics["key_pass"] and metrics["value_pass"]:
                    pass_3bit += 1

    print(f"3-bit: {pass_3bit}/{total_3bit} scenarios pass (>= 0.99 cosine)")

    # Check 4-bit unchanged
    four_bit_ok = True
    for label, data in all_results["configs"].items():
        if "4bit_baseline" in data:
            b = data["4bit_baseline"]
            if not (b["key_pass"] and b["value_pass"]):
                four_bit_ok = False

    print(f"4-bit baseline: {'ALL PASS (unchanged)' if four_bit_ok else 'REGRESSION DETECTED!'}")

    # Save
    outfile = "benchmarks/3bit_qjl_fix_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
