"""
test_cpu_load_and_forward.py — Phase 4: CPU load & forward with per-layer NaN detection.

The most important diagnostic test. If this passes, the model is correctly loaded
and inference is numerically stable on CPU. GPU issues can then be isolated separately.

Key diagnostic: test_single_forward_layer_by_layer_no_nan — attaches hooks to every
layer and reports the FIRST layer that produces NaN/Inf, so you can immediately see
where inference breaks down.

    pytest proofofconcept/tests/test_cpu_load_and_forward.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct -s

Use -s to see per-layer RMS values even on pass.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

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
def loaded_model(model_path):
    """Load CompressedChatModel on CPU and return (cm, tokenizer)."""
    from transformers import AutoTokenizer
    from chat import CompressedChatModel

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cm = CompressedChatModel(str(model_path), device='cpu', compression_mode='lossless')
    result = cm.load()
    if result is None:
        pytest.skip("Model failed to load (check cache)")
    cm.model.eval()
    return cm, tok


@pytest.fixture(scope="session")
def prompt_ids(loaded_model):
    cm, tok = loaded_model
    msgs = [{"role": "user", "content": "Write a haiku about compression"}]
    enc = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(enc, 'input_ids'):
        return enc.input_ids
    if isinstance(enc, dict):
        return enc['input_ids']
    return enc


@pytest.fixture(scope="session")
def layer_forward_results(loaded_model, prompt_ids):
    """
    Run a single forward pass with hooks on every layer.
    Returns dict: layer_name → {'nan': bool, 'inf': bool, 'rms': float, 'shape': tuple}
    """
    cm, _ = loaded_model
    model = cm.model
    results = {}
    hooks = []

    def _get_inner(m):
        inner = m
        for attr in ('model', 'language_model'):
            child = getattr(inner, attr, None)
            if child is not None:
                inner = child
        return inner

    def make_hook(layer_name):
        def fn(module, inp, output):
            val = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(val, torch.Tensor):
                vf = val.detach().float()
                results[layer_name] = {
                    'nan': bool(torch.isnan(vf).any()),
                    'inf': bool(torch.isinf(vf).any()),
                    'rms': float(vf.pow(2).mean().sqrt()),
                    'shape': tuple(vf.shape),
                }
        return fn

    inner = _get_inner(model)

    embed = getattr(inner, 'embed_tokens', None)
    if embed is not None:
        hooks.append(embed.register_forward_hook(make_hook('embed_tokens')))

    for i, layer in enumerate(getattr(inner, 'layers', [])):
        hooks.append(layer.register_forward_hook(make_hook(f'layer_{i:02d}')))

    norm = getattr(inner, 'norm', None)
    if norm is not None:
        hooks.append(norm.register_forward_hook(make_hook('final_norm')))

    lm_head = getattr(model, 'lm_head', None)
    if lm_head is not None:
        hooks.append(lm_head.register_forward_hook(make_hook('lm_head')))

    with torch.no_grad():
        model(prompt_ids)

    for h in hooks:
        h.remove()

    return results


# ── tests ─────────────────────────────────────────────────────────────────────

class TestModelLoad:
    def test_model_loads_without_exception(self, loaded_model):
        cm, _ = loaded_model
        assert cm.model is not None

    def test_compressed_module_count_reasonable(self, loaded_model):
        from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
        cm, _ = loaded_model
        n_linear = sum(1 for _, m in cm.model.named_modules()
                       if isinstance(m, AdaptiveCodebookLinear))
        n_embed = sum(1 for _, m in cm.model.named_modules()
                      if isinstance(m, AdaptiveCodebookEmbedding))
        print(f"\n  AdaptiveCodebookLinear: {n_linear}")
        print(f"  AdaptiveCodebookEmbedding: {n_embed}")
        assert n_linear >= 50, \
            f"Only {n_linear} compressed linear layers — module replacement likely failed"
        assert n_embed >= 1, "No compressed embedding layer found"

    def test_no_nan_in_parameters_after_load(self, loaded_model):
        cm, _ = loaded_model
        nan_params = []
        for name, param in cm.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append((name, param.shape))
        if nan_params:
            print(f"\n  NaN parameters ({len(nan_params)}):")
            for n, s in nan_params[:10]:
                print(f"    {n}: {s}")
        assert not nan_params, \
            f"{len(nan_params)} parameters contain NaN after load"

    def test_no_nan_in_buffers_after_load(self, loaded_model):
        cm, _ = loaded_model
        nan_bufs = []
        for name, buf in cm.model.named_buffers():
            if buf is not None and torch.isnan(buf).any():
                nan_bufs.append((name, buf.shape))
        if nan_bufs:
            print(f"\n  NaN buffers ({len(nan_bufs)}):")
            for n, s in nan_bufs[:10]:
                print(f"    {n}: {s}")
        assert not nan_bufs, f"{len(nan_bufs)} buffers contain NaN after load"

    def test_no_inf_in_parameters_after_load(self, loaded_model):
        cm, _ = loaded_model
        inf_params = []
        for name, param in cm.model.named_parameters():
            if torch.isinf(param).any():
                inf_params.append((name, param.shape))
        if inf_params:
            print(f"\n  Inf parameters ({len(inf_params)}):")
            for n, s in inf_params[:5]:
                print(f"    {n}: {s}")
        assert not inf_params, f"{len(inf_params)} parameters contain Inf after load"

    def test_codebooks_are_finite_and_nonzero(self, loaded_model):
        from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
        cm, _ = loaded_model
        bad = []
        for name, mod in cm.model.named_modules():
            if not isinstance(mod, (AdaptiveCodebookLinear, AdaptiveCodebookEmbedding)):
                continue
            if mod.mode != 'direct_codebook':
                continue
            cb = mod.codebook
            if cb is None:
                bad.append((name, "codebook is None"))
            elif not torch.isfinite(cb).all():
                bad.append((name, "codebook has NaN/Inf"))
            elif cb.abs().max() < 1e-9:
                bad.append((name, "codebook is all zeros"))
        if bad:
            for n, msg in bad[:10]:
                print(f"\n  ❌ {n}: {msg}")
        assert not bad, f"{len(bad)} modules have bad codebooks"


class TestForwardPass:
    def test_single_forward_no_exception(self, loaded_model, prompt_ids):
        cm, _ = loaded_model
        with torch.no_grad():
            out = cm.model(prompt_ids)
        assert out is not None

    def test_single_forward_logits_finite(self, loaded_model, prompt_ids):
        cm, _ = loaded_model
        with torch.no_grad():
            out = cm.model(prompt_ids)
        logits = out.logits[0, -1].float()
        nan_count = int(torch.isnan(logits).sum())
        inf_count = int(torch.isinf(logits).sum())
        assert nan_count == 0, f"Logits have {nan_count} NaN values"
        assert inf_count == 0, f"Logits have {inf_count} Inf values"

    def test_logit_std_reasonable(self, loaded_model, prompt_ids):
        cm, _ = loaded_model
        with torch.no_grad():
            out = cm.model(prompt_ids)
        logits = out.logits[0, -1].float()
        std = logits.std().item()
        mean_abs = logits.abs().mean().item()
        print(f"\n  Logit std={std:.4f}, mean_abs={mean_abs:.4f}")
        assert std > 0.01, f"Logit std suspiciously small: {std:.6f}"
        assert std > mean_abs * 0.001, f"Logit std/mean ratio too small: {std/mean_abs:.6f}"

    def test_top_token_not_pad(self, loaded_model, prompt_ids):
        cm, tok = loaded_model
        with torch.no_grad():
            out = cm.model(prompt_ids)
        top_id = int(out.logits[0, -1].argmax())
        pad_id = tok.pad_token_id
        assert top_id != pad_id, \
            f"Model predicts pad_token ({pad_id}) — likely degenerate"


class TestLayerByLayerNaN:
    """The key diagnostic test suite. Identifies the FIRST layer that produces NaN."""

    def test_single_forward_layer_by_layer_no_nan(self, layer_forward_results):
        """
        Checks every hooked layer for NaN/Inf in its output.
        On failure: prints the complete per-layer table so you can see exactly
        where inference breaks down.
        """
        sorted_layers = _sort_layer_names(list(layer_forward_results.keys()))
        first_bad = None
        print(f"\n  {'Layer':<20s}  {'RMS':>10s}  {'Shape':<25s}  Status")
        print(f"  {'-'*70}")
        for name in sorted_layers:
            r = layer_forward_results[name]
            status = "✅" if not r['nan'] and not r['inf'] else "❌ NaN" if r['nan'] else "❌ Inf"
            print(f"  {name:<20s}  {r['rms']:10.4f}  {str(r['shape']):<25s}  {status}")
            if first_bad is None and (r['nan'] or r['inf']):
                first_bad = name

        if first_bad is not None:
            pytest.fail(
                f"First NaN/Inf at layer '{first_bad}' — "
                f"check compression data for that layer's weight matrices"
            )

    def test_no_rms_explosion(self, layer_forward_results):
        """No layer's RMS should be more than 100× the previous layer's RMS."""
        sorted_layers = _sort_layer_names(list(layer_forward_results.keys()))
        rms_vals = [(n, layer_forward_results[n]['rms']) for n in sorted_layers]
        for i in range(1, len(rms_vals)):
            prev_name, prev_rms = rms_vals[i - 1]
            curr_name, curr_rms = rms_vals[i]
            if prev_rms < 1e-9:
                continue  # avoid divide-by-zero
            ratio = curr_rms / prev_rms
            if ratio > 100.0:
                pytest.fail(
                    f"RMS explosion between '{prev_name}' ({prev_rms:.4f}) "
                    f"and '{curr_name}' ({curr_rms:.4f}): ratio={ratio:.1f}×"
                )

    def test_embedding_rms_in_range(self, layer_forward_results):
        r = layer_forward_results.get('embed_tokens')
        if r is None:
            pytest.skip("embed_tokens not hooked")
        rms = r['rms']
        print(f"\n  embed_tokens RMS: {rms:.4f}")
        assert not r['nan'], "embed_tokens output contains NaN"
        assert not r['inf'], "embed_tokens output contains Inf"
        assert rms > 1e-4, f"embed_tokens RMS near zero: {rms:.6f}"

    def test_final_hidden_state_finite_and_nonzero(self, layer_forward_results):
        r = layer_forward_results.get('final_norm')
        if r is None:
            # Try last layer
            layers = [k for k in layer_forward_results if k.startswith('layer_')]
            if not layers:
                pytest.skip("No layer hooks found")
            r = layer_forward_results[sorted(layers)[-1]]
        assert not r['nan'], "Final hidden state contains NaN"
        assert not r['inf'], "Final hidden state contains Inf"
        assert r['rms'] > 1e-4, f"Final hidden state RMS near zero: {r['rms']:.6f}"


class TestGreedyGeneration:
    def test_greedy_6_tokens_diverse(self, loaded_model, prompt_ids):
        cm, _ = loaded_model
        ids = prompt_ids.clone()
        generated = []
        with torch.no_grad():
            for _ in range(6):
                out = cm.model(ids)
                next_id = int(out.logits[0, -1].argmax())
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
        unique = len(set(generated))
        print(f"\n  Generated: {generated} (unique={unique})")
        assert unique > 1, \
            f"Model stuck: repeated token {generated[0]} for all 6 steps"

    def test_generated_text_not_empty(self, loaded_model, prompt_ids):
        cm, tok = loaded_model
        with torch.no_grad():
            out = cm.model.generate(
                prompt_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        generated_ids = out[0, prompt_ids.shape[1]:]
        text = tok.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"\n  Generated: '{text}'")
        assert len(text) > 0, "Generated only special tokens — model output is degenerate"


# ── helpers ───────────────────────────────────────────────────────────────────

def _sort_layer_names(names):
    """Sort layer names: embed_tokens, layer_00..layer_N, final_norm, lm_head."""
    def key(n):
        if n == 'embed_tokens':
            return (0, 0)
        if n.startswith('layer_'):
            try:
                return (1, int(n.split('_')[1]))
            except (IndexError, ValueError):
                return (1, 999)
        if n == 'final_norm':
            return (2, 0)
        if n == 'lm_head':
            return (3, 0)
        return (4, 0)
    return sorted(names, key=key)
