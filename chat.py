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
from memory_utils import get_memory_stats, format_memory_stats


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
    ):
        self.model_path = Path(model_path).expanduser()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.compression_mode = compression_mode
        self.force_rebuild = force_rebuild
        self.use_compressed_modules = use_compressed_modules
        self.codebook_threshold = codebook_threshold
        self.use_mmap = use_mmap
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> Optional["CompressedChatModel"]:
        """Load compressed model.  Returns self on success, None on failure."""
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
        cache_tensors = self.model_path / 'codebook' / 'tensors'
        if not cache_tensors.exists() or not list(cache_tensors.glob('*.npz')):
            print(f"\nError: No compression cache at {cache_tensors}")
            print(f"Run: python proofofconcept/compress.py {self.model_path}")
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
        print("\nLoading compressed weights...")
        mse_target = (1.0 - self.codebook_threshold / 100.0) ** 2
        compressor = AdaptiveCompressor(
            self.model_path,
            compression_mode=self.compression_mode,
            store_in_model=True,
            force_rebuild=self.force_rebuild,
            mse_threshold=mse_target,
        )
        _, metadata = compressor.load_compressed(load_tensors=False)

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
        # Propagate device after possible CPU fallback inside loader.
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
                    return_tensors="pt"
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

    def generate(self, messages, max_tokens: int = 100, temperature: float = 0.7) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        input_ids = self._tokenize_messages(messages).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.time() - t0
        generated = outputs[0, input_ids.shape[1]:]
        tps = len(generated) / elapsed if elapsed > 0 else 0
        print(f"\nGeneration: {tps:.1f} tok/s ({len(generated)} tokens in {elapsed:.1f}s)")
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def chat_loop(self):
        print("=" * 70)
        print("CHAT INTERFACE (Commands: /clear, /quit)")
        print("=" * 70)
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
                    print("Cleared.")
                    continue
                messages.append({"role": "user", "content": user_input})
                print("Assistant: ", end='', flush=True)
                response = self.generate(messages, max_tokens=512)
                print(response)
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
            config, trust_remote_code=True, dtype=model_dtype
        )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Chat with a compressed model')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', default='balanced')
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--prompt', default=None, help='Single prompt (non-interactive)')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    chat_model = CompressedChatModel(
        args.model_path,
        device=args.device,
        compression_mode=args.mode,
        force_rebuild=args.force_rebuild,
    )
    if chat_model.load():
        if args.prompt:
            response = chat_model.generate(
                args.prompt, max_tokens=args.max_tokens, temperature=args.temperature
            )
            print(f"\nResponse: {response}")
        else:
            chat_model.chat_loop()


if __name__ == '__main__':
    main()
