#!/usr/bin/env python3
"""
TurboQuant CLI: Run inference with quantized KV cache.

Usage:
    # Interactive chat with 4-bit KV cache
    python -m turboquant.cli --model Qwen/Qwen2.5-3B-Instruct --kv-quant turbo_4bit

    # Single prompt
    python -m turboquant.cli --model Qwen/Qwen2.5-0.5B --kv-quant turbo_4bit \
        --prompt "What is the Pythagorean theorem?"

    # Compare FP16 vs quantized KV
    python -m turboquant.cli --model Qwen/Qwen2.5-0.5B --kv-quant turbo_4bit --compare

    # Benchmark KV quality on WikiText-2
    python -m turboquant.cli --model Qwen/Qwen2.5-0.5B --kv-quant turbo_4bit --benchmark

    # Run with Bonsai via llama-cli (GGUF models)
    python -m turboquant.cli --gguf /path/to/Bonsai-8B.gguf --kv-quant q4_0

Flags:
    --kv-quant    KV cache quantization: turbo_4bit, turbo_3bit, kivi_2bit, onebit, q8_0, q4_0, f16
    --compare     Compare FP16 vs quantized generation side-by-side
    --benchmark   Run quality benchmark on WikiText-2
    --max-tokens  Max tokens to generate (default: 128)
"""

import argparse
import sys
import os
import time
import subprocess
import json

import torch


def get_quantizer(kv_quant: str, num_heads: int, num_kv_heads: int, head_dim: int):
    """Create TurboQuant quantizer from CLI flag."""
    from turboquant.config import TurboQuantConfig
    from turboquant.core.quantizer import TurboQuantizer

    presets = {
        "turbo_4bit": TurboQuantConfig.turbo_4bit,
        "turbo_3bit": TurboQuantConfig.turbo_3bit,
        "kivi_2bit": TurboQuantConfig.kivi_2bit,
        "onebit": TurboQuantConfig.onebit_extreme,
    }

    if kv_quant not in presets:
        print(f"Error: --kv-quant must be one of: {', '.join(presets.keys())}")
        print(f"  For GGUF models, use: f16, q8_0, q4_0 (passed to llama-cli)")
        sys.exit(1)

    config = presets[kv_quant](
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    return TurboQuantizer(config), config


def _iter_kv(past_kv):
    if hasattr(past_kv, "key_cache"):
        for i in range(len(past_kv.key_cache)):
            yield past_kv.key_cache[i], past_kv.value_cache[i]
    else:
        yield from past_kv


@torch.no_grad()
def generate_with_quantized_kv(model, tokenizer, quantizer, prompt, max_tokens=128):
    """Generate text with TurboQuant KV cache quantization."""
    from transformers.cache_utils import DynamicCache

    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"].clone()
    past_kv = None

    start = time.perf_counter()
    for step in range(max_tokens):
        if past_kv is None:
            out = model(generated, use_cache=True)
        else:
            out = model(generated[:, -1:], past_key_values=past_kv, use_cache=True)

        # Quantize KV cache
        raw_kv = out.past_key_values
        qcache = DynamicCache()
        for li, (keys, values) in enumerate(_iter_kv(raw_kv)):
            k = keys.transpose(1, 2).float()
            v = values.transpose(1, 2).float()
            rk = quantizer.decode_keys(*quantizer.encode_keys(k)).transpose(1, 2).to(keys.dtype)
            rv = quantizer.decode_values(*quantizer.encode_values(v)).transpose(1, 2).to(values.dtype)
            qcache.update(rk, rv, li)
        past_kv = qcache

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        # Stream output
        token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(token_str, end="", flush=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - start
    total_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
    print(f"\n\n[{total_tokens} tokens, {total_tokens/elapsed:.1f} tok/s, KV: {quantizer.config.kv_bits}-bit]")

    return tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.no_grad()
def compare_generation(model, tokenizer, quantizer, prompt, max_tokens=64):
    """Side-by-side comparison: FP16 vs quantized KV cache."""
    from transformers.cache_utils import DynamicCache

    inputs = tokenizer(prompt, return_tensors="pt")

    # FP16 baseline
    out_fp16 = model.generate(inputs["input_ids"], max_new_tokens=max_tokens, do_sample=False)
    text_fp16 = tokenizer.decode(out_fp16[0], skip_special_tokens=True)

    # Quantized KV
    generated = inputs["input_ids"].clone()
    past_kv = None
    for _ in range(max_tokens):
        if past_kv is None:
            out = model(generated, use_cache=True)
        else:
            out = model(generated[:, -1:], past_key_values=past_kv, use_cache=True)
        raw_kv = out.past_key_values
        qcache = DynamicCache()
        for li, (keys, values) in enumerate(_iter_kv(raw_kv)):
            k = keys.transpose(1, 2).float()
            v = values.transpose(1, 2).float()
            rk = quantizer.decode_keys(*quantizer.encode_keys(k)).transpose(1, 2).to(keys.dtype)
            rv = quantizer.decode_values(*quantizer.encode_values(v)).transpose(1, 2).to(values.dtype)
            qcache.update(rk, rv, li)
        past_kv = qcache
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, nt], dim=-1)
        if nt.item() == tokenizer.eos_token_id:
            break

    text_quant = tokenizer.decode(generated[0], skip_special_tokens=True)

    # Token agreement
    t_fp16 = out_fp16[0].tolist()
    t_quant = generated[0].tolist()
    n = min(len(t_fp16), len(t_quant))
    agreement = sum(a == b for a, b in zip(t_fp16[:n], t_quant[:n])) / max(n, 1)

    print(f"  FP16 KV:  {text_fp16}")
    print(f"  {quantizer.config.kv_bits}-bit KV: {text_quant}")
    print(f"  Token agreement: {agreement:.0%}")


def run_gguf(gguf_path, kv_type, prompt, max_tokens=128):
    """Run GGUF model via llama-cli with specified KV cache type."""
    bonsai_dir = "/Users/pradip/Desktop/Learning/Claude/PrismML/Bonsai-demo"
    llama_cli = os.path.join(bonsai_dir, "bin", "mac", "llama-cli")

    if not os.path.exists(llama_cli):
        print(f"Error: llama-cli not found at {llama_cli}")
        sys.exit(1)

    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    cmd = [
        llama_cli, "-m", gguf_path,
        "-ngl", "999", "-fa", "1",
        "-c", "4096",
        "-ctk", kv_type, "-ctv", kv_type,
        "-n", str(max_tokens),
        "-p", full_prompt,
        "--temp", "0",
        "--single-turn",
        "--no-display-prompt",
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.join(bonsai_dir, "bin", "mac")

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    elapsed = time.perf_counter() - start

    # Clean output: strip banner, ANSI codes, loading noise
    import re
    output = result.stdout
    output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    output = re.sub(r'\[ Prompt:.*?]', '', output)
    # Remove everything before the assistant's actual answer
    # The answer comes after the last "| " or "|-\" spinner character
    lines = output.split("\n")
    clean_lines = []
    found_answer = False
    for line in lines:
        stripped = line.strip()
        # Skip banner, loading, metadata, empty, commands
        if any(x in stripped for x in ["▄", "▀", "█", "░", "▓", "▒",
               "Loading model", "Exiting", "build      :", "model      :",
               "modalities :", "available commands:", "/exit", "/regen",
               "/clear", "/read", "<|im_start|>", "<|im_end|>", "|-\\",
               "|/", "|-", "\\|"]):
            continue
        if stripped.startswith(">"):
            continue
        if not stripped:
            if not found_answer:
                continue
        if stripped:
            found_answer = True
        if found_answer:
            # Strip leading spinner chars
            cleaned = re.sub(r'^[\|\\\-/]+\s*', '', stripped)
            if cleaned:
                clean_lines.append(cleaned)
    output = "\n".join(clean_lines).strip()

    # Parse throughput
    full = result.stdout + "\n" + result.stderr
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", full)
    tps_info = ""
    if m:
        tps_info = f", {m.group(1)} pp tok/s, {m.group(2)} gen tok/s"

    print(output)
    print(f"\n[{elapsed:.1f}s, KV: {kv_type}{tps_info}]")


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant: Inference with quantized KV cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--gguf", type=str, help="Path to GGUF model (uses llama-cli)")
    parser.add_argument("--kv-quant", type=str, default="turbo_4bit",
                        help="KV cache quantization: turbo_4bit, turbo_3bit, kivi_2bit, onebit, f16, q8_0, q4_0")
    parser.add_argument("--prompt", type=str, help="Single prompt (otherwise interactive)")
    parser.add_argument("--compare", action="store_true", help="Compare FP16 vs quantized")
    parser.add_argument("--benchmark", action="store_true", help="Run quality benchmark")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    if not args.model and not args.gguf:
        parser.print_help()
        print("\nError: Specify --model (HuggingFace) or --gguf (GGUF file)")
        sys.exit(1)

    # ── GGUF path: use llama-cli directly ──
    if args.gguf:
        kv_type = args.kv_quant if args.kv_quant in ("f16", "q8_0", "q4_0") else "q4_0"
        if args.prompt:
            run_gguf(args.gguf, kv_type, args.prompt, args.max_tokens)
        else:
            print(f"TurboQuant CLI (GGUF mode, KV: {kv_type})")
            print("Type your prompt (Ctrl+C to exit):\n")
            while True:
                try:
                    prompt = input("> ")
                    if prompt.strip():
                        run_gguf(args.gguf, kv_type, prompt, args.max_tokens)
                        print()
                except (KeyboardInterrupt, EOFError):
                    print("\nBye!")
                    break
        return

    # ── HuggingFace model: use TurboQuant ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mc = model.config
    num_heads = mc.num_attention_heads
    num_kv_heads = getattr(mc, "num_key_value_heads", num_heads)
    head_dim = mc.hidden_size // num_heads

    quantizer, config = get_quantizer(args.kv_quant, num_heads, num_kv_heads, head_dim)
    print(f"Model: {args.model} ({mc.num_hidden_layers} layers, {num_kv_heads} KV heads, dim {head_dim})")
    print(f"KV cache: {args.kv_quant} ({config.kv_bits}-bit, {config.effective_key_bits}b keys)")

    if args.benchmark:
        from benchmarks.real_model_eval import eval_hf_model
        eval_hf_model(args.model, dtype=torch.float16)
        return

    if args.compare:
        prompts = [
            "The Pythagorean theorem states that",
            "The capital of France is Paris, which is known for",
        ]
        for p in prompts:
            print(f"\nPrompt: \"{p}\"")
            compare_generation(model, tokenizer, quantizer, p, args.max_tokens)
        return

    if args.prompt:
        generate_with_quantized_kv(model, tokenizer, quantizer, args.prompt, args.max_tokens)
        return

    # Interactive mode
    print(f"\nTurboQuant CLI (KV: {args.kv_quant})")
    print("Type your prompt (Ctrl+C to exit):\n")
    while True:
        try:
            prompt = input("> ")
            if prompt.strip():
                generate_with_quantized_kv(model, tokenizer, quantizer, prompt, args.max_tokens)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
