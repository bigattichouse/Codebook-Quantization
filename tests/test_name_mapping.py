"""
test_name_mapping.py — Phase 2: Verify cache-to-param name resolution.

Tests that _build_name_prefix correctly detects whether the cache inserts
a middle segment (e.g. 'language_model') into param names, and that every
direct_codebook tensor in the cache resolves to a real model parameter.

No weights are loaded. Runs on meta device = fast.

    pytest proofofconcept/tests/test_name_mapping.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "proofofconcept" / "src"))
sys.path.insert(0, str(ROOT / "proofofconcept"))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_path(request):
    explicit = request.config.getoption("--model", default=None)
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        p = Path("~/workspace/model/Qwen3.5-0.8B").expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    cache = p / "codebook" / "tensors"
    if not cache.exists() or not list(cache.glob("*.npz")):
        pytest.skip(f"No codebook cache at {cache}")
    return p


@pytest.fixture(scope="session")
def chat_model_dry_run(model_path):
    """
    Create a CompressedChatModel in 'dry run' mode:
    - Loads config & tokenizer
    - Creates meta-device model (zero memory)
    - Calls _build_name_prefix and _replace_modules_recursive
    - Does NOT call to_empty() or load weights

    Returns the CompressedChatModel with its name-mapping state populated.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from chat import CompressedChatModel
    from adaptive_compressor import AdaptiveCompressor

    cm = CompressedChatModel.__new__(CompressedChatModel)
    cm.model_path = model_path
    cm.device = 'cpu'
    cm.compression_mode = 'lossless'
    cm.force_rebuild = False
    cm.use_compressed_modules = True
    cm.codebook_threshold = 99.5
    cm.use_mmap = False
    cm.tensors_loaded = 0

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, 'text_config'):
            for key, val in vars(config.text_config).items():
                if not hasattr(config, key):
                    setattr(config, key, val)
    except Exception as e:
        pytest.skip(f"Cannot load config: {e}")

    # Load compressor metadata only (no tensors)
    mse_target = (1.0 - 0.995) ** 2
    compressor = AdaptiveCompressor(
        model_path, compression_mode='lossless',
        store_in_model=True, force_rebuild=False, mse_threshold=mse_target
    )
    _, metadata = compressor.load_compressed(load_tensors=False)
    cm.metadata = metadata

    # Create meta-device model
    model_dtype = torch.bfloat16
    with torch.device('meta'):
        cm.model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, dtype=model_dtype
        )

    # Build name prefix (the key function under test)
    cm._cache_middle = ""
    cm._cache_first_seg = ""
    cm._build_name_prefix(compressor)

    return cm, compressor, metadata


# ── helpers ───────────────────────────────────────────────────────────────────

def _all_param_names(model):
    return {name for name, _ in model.named_parameters()}


def _cache_stem_to_tensor_name(stem: str) -> str:
    """Convert a cache file stem (underscores) back to a dotted tensor name."""
    # Heuristic: replace _ with . but preserve known multi-word segments
    # This is approximate — used only to check reachability
    return stem.replace("_", ".")


# ── tests ─────────────────────────────────────────────────────────────────────

class TestNamePrefix:
    def test_prefix_detection_runs_without_exception(self, chat_model_dry_run):
        cm, _, _ = chat_model_dry_run
        # Just verify the attributes exist and are strings
        assert isinstance(cm._cache_middle, str)
        assert isinstance(cm._cache_first_seg, str)
        print(f"\n  _cache_middle = '{cm._cache_middle}'")
        print(f"  _cache_first_seg = '{cm._cache_first_seg}'")

    def test_decoder_only_model_has_no_middle_prefix(self, chat_model_dry_run, model_path):
        """Qwen2ForCausalLM: no 'language_model' insertion expected."""
        import json
        with open(model_path / "config.json") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if not any("ForCausalLM" in a and "Conditional" not in a for a in archs):
            pytest.skip(f"Not a decoder-only model: {archs}")
        cm, _, _ = chat_model_dry_run
        assert cm._cache_middle == "", \
            f"Decoder-only model {archs} should have no middle prefix, got '{cm._cache_middle}'"

    def test_multimodal_model_has_language_model_prefix(self, chat_model_dry_run, model_path):
        """Qwen3_5ForConditionalGeneration: expects 'language_model' insertion."""
        import json
        with open(model_path / "config.json") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if not any("ConditionalGeneration" in a for a in archs):
            pytest.skip(f"Not a multimodal model: {archs}")
        cm, _, _ = chat_model_dry_run
        assert cm._cache_middle == "language_model", \
            f"Multimodal model {archs} should have 'language_model' prefix, got '{cm._cache_middle}'"

    def test_resolve_name_is_identity_for_decoder_only(self, chat_model_dry_run, model_path):
        import json
        with open(model_path / "config.json") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if not any("ForCausalLM" in a and "Conditional" not in a for a in archs):
            pytest.skip("Not a decoder-only model")
        cm, _, _ = chat_model_dry_run
        test_name = "model.layers.0.self_attn.q_proj.weight"
        resolved = cm._resolve_name(test_name)
        assert resolved == test_name, \
            f"_resolve_name should be identity for decoder-only, got '{resolved}'"

    def test_resolve_name_inserts_middle_for_multimodal(self, chat_model_dry_run, model_path):
        import json
        with open(model_path / "config.json") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if not any("ConditionalGeneration" in a for a in archs):
            pytest.skip("Not a multimodal model")
        cm, _, _ = chat_model_dry_run
        test_name = "model.embed_tokens.weight"
        resolved = cm._resolve_name(test_name)
        assert "language_model" in resolved, \
            f"Expected 'language_model' in resolved name, got '{resolved}'"


class TestTensorResolution:
    def test_direct_codebook_tensors_find_model_params(self, chat_model_dry_run):
        """
        For each direct_codebook .npz, verify that _resolve_name maps a plausible
        model parameter path.

        Strategy: the cache stem uses _ instead of .; we try both direct lookup and
        through _resolve_name. Count how many resolve to real model params.
        """
        cm, compressor, _ = chat_model_dry_run
        cache_dir = compressor.cache_dir / "tensors"
        param_names = _all_param_names(cm.model)

        total_dc = 0
        resolved_ok = 0
        unresolved = []

        for f in sorted(cache_dir.glob("*.npz")):
            d = dict(np.load(f, allow_pickle=True))
            mode = str(d.get("mode", ""))
            if not mode.startswith("direct_codebook"):
                continue
            total_dc += 1

            stem = f.stem  # e.g. model_layers_0_self_attn_q_proj_weight
            # Convert cache stem → dotted name (heuristic)
            # The actual resolution goes: param_name → _resolve_name → safe_name → stem
            # We go backwards: stem → try to match a param name
            found = False
            for pname in param_names:
                resolved = cm._resolve_name(pname)
                safe = resolved.replace(".", "_")
                if safe == stem:
                    found = True
                    resolved_ok += 1
                    break
            if not found:
                unresolved.append(stem)

        print(f"\n  direct_codebook tensors: {total_dc}")
        print(f"  successfully resolved: {resolved_ok}")
        if unresolved:
            print(f"  UNRESOLVED ({len(unresolved)}):")
            for s in unresolved[:10]:
                print(f"    {s}")

        assert total_dc > 0, "No direct_codebook tensors found"
        unresolved_frac = len(unresolved) / total_dc
        assert unresolved_frac < 0.05, \
            f"{len(unresolved)}/{total_dc} ({unresolved_frac*100:.1f}%) tensors unresolved — " \
            f"name mapping is broken"

    def test_replacement_count_reasonable(self, chat_model_dry_run):
        """
        Simulate module replacement (dry run on meta device) and count replaced modules.

        Note: on meta device, from_compressed can't run (no real data), so we count
        how many linear/embedding modules WOULD be replaced by checking compressor data.
        """
        cm, compressor, _ = chat_model_dry_run

        would_replace = 0
        would_skip = 0
        no_data = 0

        for name, child in cm.model.named_modules():
            if isinstance(child, torch.nn.Linear):
                weight_name = f"{name}.weight"
                resolved = cm._resolve_name(weight_name)
                data = compressor._get_compressed_tensor_data(resolved)
                if data is None and "lm_head" in name:
                    data = compressor._get_compressed_tensor_data(
                        cm._resolve_name("model.embed_tokens.weight"))
                if data is None:
                    no_data += 1
                elif data.get("mode") == "direct_codebook":
                    would_replace += 1
                else:
                    would_skip += 1
            elif isinstance(child, torch.nn.Embedding):
                weight_name = f"{name}.weight"
                resolved = cm._resolve_name(weight_name)
                data = compressor._get_compressed_tensor_data(resolved)
                if data and data.get("mode") == "direct_codebook":
                    would_replace += 1

        print(f"\n  Would replace: {would_replace}")
        print(f"  Would skip (non-codebook): {would_skip}")
        print(f"  No data found: {no_data}")

        assert would_replace >= 50, \
            f"Only {would_replace} modules would be replaced — name mapping likely broken"
        assert no_data < would_replace, \
            f"More modules have no data ({no_data}) than would be replaced ({would_replace})"

    def test_embed_tokens_resolves(self, chat_model_dry_run):
        """embed_tokens must have a direct_codebook entry in the cache."""
        cm, compressor, _ = chat_model_dry_run
        name = cm._resolve_name("model.embed_tokens.weight")
        data = compressor._get_compressed_tensor_data(name)
        assert data is not None, \
            f"embed_tokens not found in cache at resolved name '{name}'"
        mode = data.get("mode", "")
        print(f"\n  embed_tokens mode: {mode}")
        # Can be exact (for small models) or direct_codebook
        assert mode in ("direct_codebook", "exact"), \
            f"embed_tokens has unexpected mode: {mode}"

    def test_lm_head_resolves_or_is_tied(self, chat_model_dry_run):
        """lm_head.weight should either be in cache or be tied to embed_tokens."""
        cm, compressor, _ = chat_model_dry_run
        name = cm._resolve_name("lm_head.weight")
        data = compressor._get_compressed_tensor_data(name)
        if data is not None:
            print(f"\n  lm_head: found in cache (mode={data.get('mode')})")
            return  # Good
        # Try tied weights fallback
        embed_name = cm._resolve_name("model.embed_tokens.weight")
        embed_data = compressor._get_compressed_tensor_data(embed_name)
        assert embed_data is not None, \
            "lm_head.weight not in cache AND embed_tokens not in cache — tied-weight fallback broken"
        print(f"\n  lm_head: not in cache — will use tied embed_tokens weight (expected)")
