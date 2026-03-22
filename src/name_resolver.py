"""
name_resolver.py — Maps model parameter names to cache tensor names.

The compression cache stores tensors under names that may differ from the
model's parameter names.  Two known mismatches are handled:

1. Inserted middle segment (Qwen3.5 multimodal):
     model param:  model.embed_tokens.weight
     cache stem:   model_language_model_embed_tokens_weight
     gap:          "language_model" inserted after "model"

2. Stripped leading segment (GPT-2 and similar):
     model param:  transformer.wte.weight
     cache stem:   wte_weight
     The safetensors file omits the top-level module prefix.

Decoder-only models (Qwen2, Llama, Mistral) have no name mismatch.

Usage:
    resolver = NameResolver.from_model_and_compressor(model, compressor)
    cache_name = resolver.resolve("model.layers.0.self_attn.q_proj.weight")
"""

from pathlib import Path


class NameResolver:
    """Translates model parameter names to compression cache tensor names."""

    def __init__(self, cache_middle: str = "", cache_first_seg: str = "",
                 strip_first_seg: bool = False):
        self._cache_middle = cache_middle
        self._cache_first_seg = cache_first_seg
        self._strip_first_seg = strip_first_seg  # GPT-2 style: drop leading segment

    @classmethod
    def from_model_and_compressor(cls, model, compressor) -> "NameResolver":
        """Auto-detect naming convention by matching param names against cache stems."""
        cache_dir = compressor.cache_dir / "tensors"
        cache_middle = ""
        cache_first_seg = ""
        strip_first_seg = False

        if not cache_dir.exists():
            return cls(cache_middle, cache_first_seg, strip_first_seg)

        cache_stems = {f.stem for f in cache_dir.glob("*.npz")}
        if not cache_stems:
            return cls(cache_middle, cache_first_seg, strip_first_seg)

        for param_name, _ in model.named_parameters():
            safe_param = param_name.replace(".", "_")
            if safe_param in cache_stems:
                break  # direct match → no gap

            parts = param_name.split(".", 1)
            if len(parts) < 2:
                continue
            first_seg = parts[0]
            rest_safe = parts[1].replace(".", "_")

            # Case 2: stripped leading segment (e.g. GPT-2's 'transformer.' prefix)
            if rest_safe in cache_stems:
                print(f"  Detected cache name convention: leading '{first_seg}.' is stripped")
                cache_first_seg = first_seg
                strip_first_seg = True
                return cls(cache_middle, cache_first_seg, strip_first_seg)

            # Case 1: inserted middle segment (e.g. Qwen multimodal)
            prefix = first_seg + "_"
            suffix = "_" + rest_safe
            for stem in cache_stems:
                if stem.startswith(prefix) and stem.endswith(suffix):
                    middle = stem[len(prefix):-len(suffix)]
                    cache_middle = middle
                    cache_first_seg = first_seg
                    print(f"  Detected cache name gap: '{first_seg}.*' → '{first_seg}.{middle}.*'")
                    return cls(cache_middle, cache_first_seg, strip_first_seg)
            break  # checked first param, no match found

        return cls(cache_middle, cache_first_seg, strip_first_seg)

    def resolve(self, param_name: str) -> str:
        """Translate a model param name to the cache tensor name."""
        parts = param_name.split(".", 1)

        if self._strip_first_seg:
            # GPT-2 style: cache has no leading segment
            if len(parts) == 2 and parts[0] == self._cache_first_seg:
                return parts[1]
            return param_name

        if self._cache_middle:
            # Qwen multimodal style: extra segment inserted
            if len(parts) == 2 and parts[0] == self._cache_first_seg:
                return f"{parts[0]}.{self._cache_middle}.{parts[1]}"

        return param_name

    def resolve_tied(self, name: str, fallback: str) -> str:
        """Resolve name; if missing, try a fallback (for tied weights like lm_head)."""
        return self.resolve(name) if name else self.resolve(fallback)

    def idx_file_for(self, tensor_name: str, model_path: Path, use_mmap: bool):
        """Return the .idx mmap path for a tensor, or None if mmap is disabled."""
        if not use_mmap:
            return None
        safe = tensor_name.replace('.', '_').replace('/', '_')
        return model_path / 'codebook' / 'tensors' / f"{safe}.idx"
