#!/usr/bin/env python3
"""
Chat with a compressed LLM — generic multi-model support.

Supports Llama 3.x, Mistral/Devstral, Gemma 3, Qwen 3/3.5, and any
HuggingFace CausalLM whose weights live in safetensors files.

This file is intentionally thin.  All heavy lifting lives in src/:
  model_loader.py  — meta-device creation, weight loading, module replacement
  name_resolver.py — cache↔param name mapping
  rope_utils.py    — RoPE inv_freq reinitialization
  memory_utils.py  — RAM/VRAM accounting
  compressed_modules.py — AdaptiveCodebookLinear / AdaptiveCodebookEmbedding
"""

import sys
import time
import argparse
import gc
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adaptive_compressor import AdaptiveCompressor
from model_loader import CompressedModelLoader
from name_resolver import NameResolver
from memory_utils import get_memory_stats, format_memory_stats, resolve_device, synchronize
from uncompressed_loader import UncompressedKernelLoader


class CompressedChatModel:
    """Compressed model wrapper: load once, generate many times."""

    def __init__(
        self,
        model_path,
        device: str = 'cuda',
        compression_mode: str = 'balanced',
        force_rebuild: bool = False,
        use_compressed_modules: bool = True,
        codebook_threshold: float = 99.5,
        use_mmap: bool = False,
        use_kv_quant: bool = False,
        enable_thinking: bool = False,
        entropy_code: bool = False,
    ):
        self.model_path = Path(model_path).expanduser()
        self.device = resolve_device(device)
        self.compression_mode = compression_mode
        self.force_rebuild = force_rebuild
        self.use_compressed_modules = use_compressed_modules
        self.codebook_threshold = codebook_threshold
        self.enable_thinking = enable_thinking
        self.use_mmap = use_mmap
        self.use_kv_quant = use_kv_quant
        self.entropy_code = entropy_code
        self.model = None
        self.tokenizer = None
        self._session_cache = None
        self._session_token_count = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> Optional["CompressedChatModel"]:
        """Load model.  Returns self on success, None on failure."""
        if self.compression_mode == 'uncompressed':
            return self._load_uncompressed()
        t0 = time.time()
        _banner("COMPRESSED MODEL LOAD")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}  |  Mode: {self.compression_mode}"
              f"  |  Threshold: {self.codebook_threshold}%")
        print("=" * 80)

        mem_before = get_memory_stats()
        print("Memory before:")
        print(format_memory_stats(mem_before))

        # -- Validate cache -----------------------------------------------
        # Build a temporary compressor just to resolve the cache directory
        # (which now includes the SNR tier and optional -huffman suffix).
        _tmp_compressor = AdaptiveCompressor(
            self.model_path, compression_mode=self.compression_mode,
            store_in_model=True, entropy_code=self.entropy_code,
        )
        cache_tensors = _tmp_compressor.cache_dir / 'tensors'
        if not cache_tensors.exists() or not list(cache_tensors.glob('*.npz')):
            print(f"\nError: No compression cache at {cache_tensors}")
            print(f"Run: python proofofconcept/compress.py {self.model_path} --mode {self.compression_mode}")
            return None

        # -- Config & tokenizer -------------------------------------------
        try:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            # Promote nested text_config keys so the rest of the code sees them at top level.
            if hasattr(config, 'text_config') and config.text_config is not None:
                for key, val in vars(config.text_config).items():
                    if not hasattr(config, key):
                        setattr(config, key, val)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as exc:
            print(f"\nError loading config/tokenizer: {exc}")
            return None

        self.vocab_size = (
            getattr(config, 'vocab_size', None) or
            getattr(getattr(config, 'text_config', None), 'vocab_size', None) or
            len(self.tokenizer)
        )

        # -- Compressor & codebooks ---------------------------------------
        print("\nLoading compressed weights (parallel preload)...")
        mse_target = (1.0 - self.codebook_threshold / 100.0) ** 2
        compressor = AdaptiveCompressor(
            self.model_path,
            compression_mode=self.compression_mode,
            store_in_model=True,
            force_rebuild=self.force_rebuild,
            mse_threshold=mse_target,
        )
        # load_tensors=True uses ThreadPoolExecutor to preload all npz files in
        # parallel.  This is much faster than sequential on-demand loading because
        # (a) I/O is parallelised and (b) each tensor is read exactly once instead
        # of twice (once in _load_exact_weights, again in _replace_modules_recursive).
        _, metadata = compressor.load_compressed(load_tensors=True)

        model_dtype = _resolve_dtype(config)

        print("  Materializing global codebooks...")
        codebooks = {}
        for ttype, cb_tensor in metadata.get('global_codebooks', {}).items():
            codebooks[ttype] = cb_tensor.to(device=self.device, dtype=torch.float32)
            print(f"    Codebook: {ttype} ({len(cb_tensor)} entries)")

        # -- Load model via CompressedModelLoader -------------------------
        loader = CompressedModelLoader(
            model_path=self.model_path,
            device=self.device,
            compressor=compressor,
            codebooks=codebooks,
            use_compressed_modules=self.use_compressed_modules,
            use_mmap=self.use_mmap,
        )
        resolver = NameResolver.from_model_and_compressor(
            # We need a minimal model for name detection — create it on meta.
            # loader.create_and_load() will recreate it; this avoids a separate
            # _build_name_prefix pass by doing a quick meta instantiation first.
            _meta_model(config, model_dtype),
            compressor,
        )

        self.model = loader.create_and_load(config, model_dtype, resolver)
        # Propagate inference device from loader.
        self.device = loader.device

        # Free the preloaded weight cache — all data is now in GPU modules or
        # exact-mode buffers.  Frees the ~15 GB of packed indices from RAM.
        if hasattr(compressor, '_loaded_weights'):
            compressor._loaded_weights = {}
        gc.collect()
        elapsed = time.time() - t0
        _banner(f"MODEL READY in {elapsed:.1f}s")
        print(format_memory_stats(get_memory_stats()))
        print("=" * 80)
        return self

    def _load_uncompressed(self) -> Optional["CompressedChatModel"]:
        """Load stock model weights and route all layers through HIP raw kernel."""
        t0 = time.time()
        _banner("UNCOMPRESSED (HIP RAW KERNEL) LOAD")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}  |  Mode: uncompressed (raw HIP matmul)")
        print("=" * 80)

        mem_before = get_memory_stats()
        print("Memory before:")
        print(format_memory_stats(mem_before))

        try:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            if hasattr(config, 'text_config') and config.text_config is not None:
                for key, val in vars(config.text_config).items():
                    if not hasattr(config, key):
                        setattr(config, key, val)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as exc:
            print(f"\nError loading config/tokenizer: {exc}")
            return None

        self.vocab_size = (
            getattr(config, 'vocab_size', None) or
            getattr(getattr(config, 'text_config', None), 'vocab_size', None) or
            len(self.tokenizer)
        )

        model_dtype = _resolve_dtype(config)

        loader = UncompressedKernelLoader(device=self.device)
        try:
            self.model = loader.load(self.model_path, config=config, dtype=model_dtype)
        except Exception as exc:
            print(f"\nError loading model: {exc}")
            return None

        self.device = loader.device
        gc.collect()

        elapsed = time.time() - t0
        _banner(f"MODEL READY in {elapsed:.1f}s")
        print(format_memory_stats(get_memory_stats()))
        print("=" * 80)
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _tokenize_messages(self, messages):
        """Tokenize chat messages, falling back to plain-text if needed."""
        if (hasattr(self.tokenizer, 'apply_chat_template') and
                self.tokenizer.chat_template is not None):
            try:
                tokenized = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", enable_thinking=self.enable_thinking
                )
                if hasattr(tokenized, 'input_ids'):
                    return tokenized.input_ids
                if isinstance(tokenized, dict):
                    return tokenized['input_ids']
                return tokenized
            except Exception:
                pass

        parts = []
        for msg in messages:
            role, content = msg.get("role", "user"), msg.get("content", "")
            if role == "system":
                parts.append(f"{content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant:")
        return self.tokenizer("".join(parts), return_tensors="pt")["input_ids"]

    def generate(self, messages, max_tokens: int = 100, temperature: float = 0.7,
                 stream: bool = False) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        input_ids = self._tokenize_messages(messages).to(self.device)

        t0 = time.time()
        base_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.use_kv_quant:
            from kv_cache_quant import INT8KVCache
            base_kwargs['past_key_values'] = INT8KVCache()

        if stream:
            import threading
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = {**base_kwargs, 'streamer': streamer}
            def _run():
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            chunks = []
            for chunk in streamer:
                print(chunk, end='', flush=True)
                chunks.append(chunk)
            thread.join()
            print()  # newline after streamed output
            full_text = "".join(chunks)
            elapsed = time.time() - t0
            n_tokens = len(self.tokenizer.encode(full_text, add_special_tokens=False))
            tps = n_tokens / elapsed if elapsed > 0 else 0
            print(f"\nGeneration: {tps:.1f} tok/s ({n_tokens} tokens in {elapsed:.1f}s)")
            return full_text

        with torch.no_grad():
            outputs = self.model.generate(**base_kwargs)

        elapsed = time.time() - t0
        generated = outputs[0, input_ids.shape[1]:]
        tps = len(generated) / elapsed if elapsed > 0 else 0
        print(f"\nGeneration: {tps:.1f} tok/s ({len(generated)} tokens in {elapsed:.1f}s)")
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Session (persistent KV cache across turns)
    # ------------------------------------------------------------------

    def _make_cache(self):
        """Create a fresh cache of the configured type."""
        if self.use_kv_quant:
            from kv_cache_quant import INT8KVCache
            return INT8KVCache()
        from transformers import DynamicCache
        return DynamicCache()

    def reset_session(self):
        self._session_cache = None
        self._session_token_count = 0

    def _session_generate(self, messages, max_tokens: int = 512) -> str:
        """Stream a response using persistent KV cache across turns.

        Only the tokens not yet in the cache are sent to the model each turn.
        The cache is mutated in-place by model.generate(), so it accumulates
        correctly across calls without needing to capture the return value.
        """
        import threading
        from transformers import TextIteratorStreamer

        full_ids = self._tokenize_messages(messages).to(self.device)

        if self._session_cache is None:
            self._session_cache = self._make_cache()
            self._session_token_count = 0

        # Only send tokens not yet processed by the model
        new_ids = full_ids[:, self._session_token_count:]
        if new_ids.shape[1] == 0:
            return ""

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=new_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            past_key_values=self._session_cache,
            use_cache=True,
            streamer=streamer,
        )

        t0 = time.time()

        def _run():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        chunks = []
        for chunk in streamer:
            print(chunk, end='', flush=True)
            chunks.append(chunk)
        thread.join()
        print()

        full_text = "".join(chunks)
        # Cache was mutated in-place — read back how many tokens it now holds
        self._session_token_count = self._session_cache.get_seq_length()

        elapsed = time.time() - t0
        n_tokens = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        tps = n_tokens / elapsed if elapsed > 0 else 0
        kv_info = ""
        if hasattr(self._session_cache, 'vram_mb'):
            kv_info = (f"  |  KV: {self._session_cache.vram_mb():.0f} MB"
                       f" ({self._session_token_count} tok cached)")
        print(f"\nGeneration: {tps:.1f} tok/s ({n_tokens} tokens in {elapsed:.1f}s){kv_info}")
        return full_text

    def chat_loop(self):
        kv_label = "INT8" if self.use_kv_quant else "bf16"
        print("=" * 70)
        print(f"CHAT INTERFACE  |  KV cache: {kv_label}  |  Commands: /clear /quit /kvstats")
        print("=" * 70)
        self.reset_session()
        messages = []
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ('/quit', '/exit', '/q'):
                    break
                if user_input.lower() == '/clear':
                    messages = []
                    self.reset_session()
                    print("Cleared.")
                    continue
                if user_input.lower() == '/kvstats':
                    if self._session_cache and hasattr(self._session_cache, 'stats'):
                        s = self._session_cache.stats()
                        print(f"  KV cache: {s['seq_len']} tokens, "
                              f"{s['quant_mb']:.0f} MB INT8 "
                              f"(bf16 equiv: {s['bf16_equiv_mb']:.0f} MB, "
                              f"{s['ratio']:.2f}× compression)")
                    elif self._session_cache:
                        print(f"  KV cache: {self._session_cache.get_seq_length()} tokens (bf16)")
                    else:
                        print("  No active session cache.")
                    continue
                messages.append({"role": "user", "content": user_input})
                print("\nAssistant: ", end='', flush=True)
                response = self._session_generate(messages, max_tokens=512)
                messages.append({"role": "assistant", "content": response})
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"Error: {exc}")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _banner(title: str):
    print(f"\n{'='*80}\n{title}\n{'='*80}")


def _resolve_dtype(config):
    """Extract model_dtype from config, defaulting to bfloat16."""
    dtype = (
        getattr(config, 'dtype', None) or
        getattr(getattr(config, 'text_config', None), 'dtype', None) or
        torch.bfloat16
    )
    if isinstance(dtype, str):
        dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(dtype, torch.bfloat16)
    return dtype


def _meta_model(config, model_dtype):
    """Create a zero-memory meta-device model for name-prefix detection."""
    from transformers import AutoModelForCausalLM
    with torch.device('meta'):
        return AutoModelForCausalLM.from_config(
            config, trust_remote_code=True
        )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _run_benchmark(chat_model):
    """Measure prefill latency and steady-state decode speed across prompt lengths."""
    import torch

    SHORT  = "Write a haiku about data compression."
    MEDIUM = "The quick brown fox jumps over the lazy dog. " * 12   # ~60 tokens
    LONG   = "The quick brown fox jumps over the lazy dog. " * 55   # ~250 tokens

    kv_label = "INT8" if chat_model.use_kv_quant else "bf16"
    print("\n" + "=" * 72)
    print("BENCHMARK — prefill latency + decode speed")
    print(f"Mode: {chat_model.compression_mode}  |  Device: {chat_model.device}  |  KV cache: {kv_label}")
    print("=" * 72)

    sync = synchronize

    def _gen(**kw):
        base = dict(do_sample=False,
                    pad_token_id=chat_model.tokenizer.pad_token_id,
                    eos_token_id=chat_model.tokenizer.eos_token_id)
        if chat_model.use_kv_quant:
            from kv_cache_quant import INT8KVCache
            base['past_key_values'] = INT8KVCache()
        base.update(kw)
        with torch.no_grad():
            return chat_model.model.generate(**base)

    results = []
    for label, prompt in [("short (~5 tok)", SHORT),
                           ("medium (~60 tok)", MEDIUM),
                           ("long (~250 tok)", LONG)]:
        messages = [{"role": "user", "content": prompt}]
        input_ids = chat_model._tokenize_messages(messages).to(chat_model.device)
        n_prompt = input_ids.shape[1]

        # --- Prefill (first token) ---
        sync()
        t0 = time.time()
        _gen(input_ids=input_ids, max_new_tokens=1)
        sync()
        prefill_s = time.time() - t0

        # --- Decode warmup (5 tokens) then measure 30 ---
        _gen(input_ids=input_ids, max_new_tokens=5)
        sync()
        t0 = time.time()
        _gen(input_ids=input_ids, max_new_tokens=30)
        sync()
        decode_tps = 30 / (time.time() - t0)

        results.append((label, n_prompt, prefill_s, decode_tps))
        print(f"  {label:18s}  prompt={n_prompt:4d} tok  "
              f"prefill={prefill_s:.2f}s  decode={decode_tps:.2f} tok/s")

    print("\n" + "=" * 72)
    print(f"{'Prompt length':<20} {'Tokens':>7} {'Prefill':>10} {'Decode':>14}")
    print("-" * 56)
    for label, n, p, d in results:
        print(f"{label:<20} {n:>7} {p:>9.2f}s {d:>12.2f} tok/s")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Chat with a compressed model')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', default='balanced',
                        choices=['balanced', 'lossless', 'uncompressed'])
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--prompt', default=None, help='Single prompt (non-interactive)')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--benchmark', action='store_true',
                        help='Run prefill + decode benchmark across prompt lengths')
    parser.add_argument('--kv-quant', action='store_true',
                        help='Store KV cache as INT8 (per-head scale); ~2× less VRAM for long context')
    parser.add_argument('--thinking', action='store_true',
                        help='Enable model chain-of-thought reasoning (Qwen3 thinking mode)')
    parser.add_argument('--entropy-code', action='store_true',
                        help='Load from a Huffman-encoded cache (created with compress.py --entropy-code)')
    args = parser.parse_args()

    chat_model = CompressedChatModel(
        args.model_path,
        device=args.device,
        compression_mode=args.mode,
        force_rebuild=args.force_rebuild,
        use_kv_quant=args.kv_quant,
        enable_thinking=args.thinking,
        entropy_code=args.entropy_code,
    )
    if chat_model.load():
        if args.benchmark:
            _run_benchmark(chat_model)
        elif args.prompt:
            response = chat_model.generate(
                args.prompt, max_tokens=args.max_tokens, temperature=args.temperature
            )
            print(f"\nResponse: {response}")
        else:
            chat_model.chat_loop()


if __name__ == '__main__':
    main()
