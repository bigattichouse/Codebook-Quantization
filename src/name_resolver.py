"""
name_resolver.py — Maps model parameter names to cache tensor names.

The compression cache stores tensors under names that may differ from the
model's parameter names by an inserted middle segment.

Example (Qwen3.5 multimodal):
  model param:  model.embed_tokens.weight
  cache stem:   model_language_model_embed_tokens_weight
  inserted:     "language_model"

Decoder-only models (Qwen2, Llama, Mistral) have no inserted segment.

Usage:
    resolver = NameResolver.from_model_and_compressor(model, compressor)
    cache_name = resolver.resolve("model.layers.0.self_attn.q_proj.weight")
"""

from pathlib import Path


class NameResolver:
    """Translates model parameter names to compression cache tensor names."""

    def __init__(self, cache_middle: str = "", cache_first_seg: str = ""):
        self._cache_middle = cache_middle
        self._cache_first_seg = cache_first_seg

    @classmethod
    def from_model_and_compressor(cls, model, compressor) -> "NameResolver":
        """Auto-detect prefix by matching param names against cache stems."""
        cache_dir = compressor.cache_dir / "tensors"
        cache_middle = ""
        cache_first_seg = ""

        if not cache_dir.exists():
            return cls(cache_middle, cache_first_seg)

        cache_stems = {f.stem for f in cache_dir.glob("*.npz")}
        if not cache_stems:
            return cls(cache_middle, cache_first_seg)

        for param_name, _ in model.named_parameters():
            safe_param = param_name.replace(".", "_")
            if safe_param in cache_stems:
                break  # direct match → no gap

            parts = param_name.split(".", 1)
            if len(parts) < 2:
                continue
            first_seg = parts[0]
            rest_safe = parts[1].replace(".", "_")
            prefix = first_seg + "_"
            suffix = "_" + rest_safe

            for stem in cache_stems:
                if stem.startswith(prefix) and stem.endswith(suffix):
                    middle = stem[len(prefix):-len(suffix)]
                    cache_middle = middle
                    cache_first_seg = first_seg
                    print(f"  Detected cache name gap: '{first_seg}.*' → '{first_seg}.{middle}.*'")
                    return cls(cache_middle, cache_first_seg)
            break  # checked first param, no match found

        return cls(cache_middle, cache_first_seg)

    def resolve(self, param_name: str) -> str:
        """Translate a model param name to the cache tensor name."""
        if not self._cache_middle:
            return param_name
        parts = param_name.split(".", 1)
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
