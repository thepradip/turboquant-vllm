"""Inference engine integrating TurboQuant with HuggingFace models.

Provides a high-level interface to run inference with quantized KV cache
on any HuggingFace causal language model. Hooks into the model's attention
layers to intercept and quantize KV cache entries.
"""

from __future__ import annotations

import torch
from typing import Optional
from tqdm import tqdm

from turboquant.config import TurboQuantConfig
from turboquant.core.quantizer import TurboQuantizer
from turboquant.core.kv_cache import QuantizedKVCache
from turboquant.engine.attention import QuantizedAttention


class TurboInferenceEngine:
    """High-level inference engine with quantized KV cache.

    Usage:
        config = TurboQuantConfig.turbo_4bit()
        engine = TurboInferenceEngine(config)
        engine.load_model("meta-llama/Llama-3.1-8B")
        output = engine.generate("Hello, world!", max_new_tokens=100)
    """

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.quantizer = TurboQuantizer(config)
        self.attention = QuantizedAttention(
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        )
        self.model = None
        self.tokenizer = None
        self.kv_cache: Optional[QuantizedKVCache] = None

    def load_model(self, model_name: str, **kwargs) -> None:
        """Load a HuggingFace model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.dtype == "float16" else torch.float32,
            device_map=self.config.device if self.config.device != "cpu" else None,
            **kwargs,
        )

        if self.config.device == "cpu":
            self.model = self.model.float()

        self.model.eval()

        # Update config from model
        model_config = self.model.config
        self.config.num_heads = getattr(model_config, "num_attention_heads", self.config.num_heads)
        self.config.num_kv_heads = getattr(
            model_config, "num_key_value_heads", self.config.num_kv_heads
        )
        self.config.head_dim = getattr(model_config, "head_dim", self.config.head_dim)

        # Reinitialize components with updated config
        self.quantizer = TurboQuantizer(self.config)
        self.attention = QuantizedAttention(
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
        )

        num_layers = getattr(model_config, "num_hidden_layers", 32)
        self.kv_cache = QuantizedKVCache(self.config, num_layers)
        self.kv_cache.set_quantizer(self.quantizer)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        compress_kv: bool = True,
    ) -> str:
        """Generate text with quantized KV cache.

        Prefills the KV cache, compresses it with TurboQuant, then generates
        tokens using the compressed cache. Compression happens once after
        prefill — generation uses the dequantized-back FP16 cache.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            compress_kv: Whether to compress KV cache after prefill

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        if compress_kv:
            # Step 1: Prefill — run forward pass to build KV cache
            outputs = self.model(input_ids, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits

            # Step 2: Compress KV cache with TurboQuant
            from turboquant.compress import compress_cache
            self._last_compress_result = compress_cache(
                cache,
                head_dim=self.config.head_dim,
                num_kv_heads=self.config.num_kv_heads,
                bits=self.config.effective_key_bits,
                device=str(self.model.device),
            )

            # Step 3: Generate tokens using compressed cache
            generated = input_ids
            for _ in range(max_new_tokens):
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                outputs = self.model(next_token, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                logits = outputs.logits

            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        else:
            # No compression — standard generation
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        texts: list[str],
        max_length: int = 512,
        batch_size: int = 1,
    ) -> dict[str, float]:
        """Evaluate perplexity on a list of texts.

        This is the primary quality metric for KV cache quantization --
        perplexity should remain close to the FP16 baseline.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating perplexity"):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
        }

    def benchmark_kv_cache(
        self,
        seq_lengths: list[int] = [256, 512, 1024, 2048, 4096],
    ) -> list[dict]:
        """Benchmark KV cache memory at various sequence lengths."""
        results = []
        for seq_len in seq_lengths:
            # Simulate KV cache entries
            batch_size = 1
            kv_cache = QuantizedKVCache(self.config, num_layers=32)
            kv_cache.set_quantizer(self.quantizer)

            # Generate random KV pairs
            for layer in range(32):
                keys = torch.randn(
                    batch_size, seq_len, self.config.num_kv_heads, self.config.head_dim
                )
                values = torch.randn(
                    batch_size, seq_len, self.config.num_kv_heads, self.config.head_dim
                )
                kv_cache.append(layer, keys, values)

            mem = kv_cache.memory_usage()

            # FP16 baseline
            fp16_bytes = (
                2 * batch_size * seq_len * self.config.num_kv_heads
                * self.config.head_dim * 2 * 32  # 2 for K+V, 2 bytes per FP16, 32 layers
            )

            results.append({
                "seq_len": seq_len,
                "quantized_mb": mem["total_mb"],
                "fp16_mb": fp16_bytes / (1024 * 1024),
                "compression_ratio": fp16_bytes / max(mem["total_bytes"], 1),
                "memory_savings_pct": (1 - mem["total_bytes"] / max(fp16_bytes, 1)) * 100,
            })

        return results
