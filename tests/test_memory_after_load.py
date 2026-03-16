"""
test_memory_after_load.py — Regression test: CPU RAM freed after model load.

Checks that the preloaded tensor cache (_loaded_weights) is cleared after
create_and_load() so compressed numpy arrays don't hold CPU RAM for the
lifetime of the process.

Run:
    pytest proofofconcept/tests/test_memory_after_load.py -v

Requires the compressed cache to exist (run compress.py first).
Skipped automatically if no cache or model is present.
"""

import gc
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

MODEL_PATH = Path(os.environ.get('COMPRESS_MODEL_PATH',
                  '~/workspace/model/Qwen3.5-9B')).expanduser()

# Find whichever compressed cache exists
def _find_cache_dir(model_path: Path):
    for d in ['codebook-30dB', 'codebook-lossless', 'codebook-25dB', 'codebook']:
        p = model_path / d / 'tensors'
        if p.exists() and list(p.glob('*.npz')):
            mode_map = {
                'codebook-30dB': 'balanced',
                'codebook-lossless': 'lossless',
                'codebook-25dB': 'lossy',
                'codebook': 'balanced',
            }
            return model_path / d, mode_map[d]
    return None, None

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Model not found at {MODEL_PATH}"
)

needs_cache = pytest.mark.skipif(
    _find_cache_dir(MODEL_PATH)[0] is None,
    reason="No compressed cache found — run compress.py first"
)


def _rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1e3
    except OSError:
        pass
    return 0.0


@needs_model
@needs_cache
def test_loaded_weights_cleared_after_load():
    """_loaded_weights must be empty after create_and_load() completes."""
    import torch
    from transformers import AutoConfig
    from adaptive_compressor import AdaptiveCompressor
    from model_loader import CompressedModelLoader
    from name_resolver import NameResolver
    from memory_utils import resolve_device

    cache_dir, mode = _find_cache_dir(MODEL_PATH)
    device = resolve_device('cuda')

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if hasattr(config, 'text_config') and config.text_config is not None:
        for k, v in vars(config.text_config).items():
            if not hasattr(config, k):
                setattr(config, k, v)

    compressor = AdaptiveCompressor(MODEL_PATH, compression_mode=mode, store_in_model=True)
    _, metadata = compressor.load_compressed(load_tensors=True)

    # Confirm tensors were preloaded
    assert hasattr(compressor, '_loaded_weights'), "_loaded_weights not created during load"
    n_preloaded = len(compressor._loaded_weights)
    assert n_preloaded > 0, "No tensors were preloaded — test is vacuous"

    codebooks = {}
    for ttype, cb in metadata.get('global_codebooks', {}).items():
        codebooks[ttype] = cb.to(device=device, dtype=torch.float32)

    from transformers import AutoModelForCausalLM
    with torch.device('meta'):
        meta = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    resolver = NameResolver.from_model_and_compressor(meta, compressor)
    del meta

    loader = CompressedModelLoader(
        model_path=MODEL_PATH, device=device,
        compressor=compressor, codebooks=codebooks,
    )
    model = loader.create_and_load(config, torch.bfloat16, resolver)

    # KEY ASSERTION: cache must be cleared after loading
    remaining = len(getattr(compressor, '_loaded_weights', {}))
    assert remaining == 0, (
        f"_loaded_weights still holds {remaining} tensors after create_and_load(). "
        f"CPU RAM is being wasted — regression in memory cleanup."
    )

    del model, loader, compressor
    gc.collect()


@needs_model
@needs_cache
def test_cpu_rss_reasonable_after_load():
    """CPU RSS after loading should not be dominated by leftover compressed data.

    Heuristic: for a ~19 GB model at ~1.5x compression (12 GB compressed),
    CPU RSS after load should be well under 8 GB once the preload cache is freed.
    Adjust MAX_RSS_MB if you use a different model.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from adaptive_compressor import AdaptiveCompressor
    from model_loader import CompressedModelLoader
    from name_resolver import NameResolver
    from memory_utils import resolve_device

    MAX_RSS_MB = 8_000  # 8 GB — conservative ceiling for post-load CPU RSS

    rss_before = _rss_mb()

    cache_dir, mode = _find_cache_dir(MODEL_PATH)
    device = resolve_device('cuda')

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if hasattr(config, 'text_config') and config.text_config is not None:
        for k, v in vars(config.text_config).items():
            if not hasattr(config, k):
                setattr(config, k, v)

    compressor = AdaptiveCompressor(MODEL_PATH, compression_mode=mode, store_in_model=True)
    _, metadata = compressor.load_compressed(load_tensors=True)

    codebooks = {}
    for ttype, cb in metadata.get('global_codebooks', {}).items():
        codebooks[ttype] = cb.to(device=device, dtype=torch.float32)

    with torch.device('meta'):
        meta = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    resolver = NameResolver.from_model_and_compressor(meta, compressor)
    del meta

    loader = CompressedModelLoader(
        model_path=MODEL_PATH, device=device,
        compressor=compressor, codebooks=codebooks,
    )
    model = loader.create_and_load(config, torch.bfloat16, resolver)

    del compressor, loader
    gc.collect()

    rss_after = _rss_mb()
    rss_delta = rss_after - rss_before

    print(f"\n  RSS before: {rss_before:.0f} MB")
    print(f"  RSS after : {rss_after:.0f} MB")
    print(f"  RSS delta : {rss_delta:.0f} MB")

    assert rss_after < MAX_RSS_MB, (
        f"CPU RSS after load is {rss_after:.0f} MB — exceeds {MAX_RSS_MB} MB ceiling. "
        f"Compressed data may not be freed after GPU upload."
    )

    del model
    gc.collect()
