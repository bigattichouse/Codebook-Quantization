"""
test_rope_initialization.py — Verify RoPE inv_freq is correctly initialized.

Root cause of layer_00 divergence on Qwen3/Llama/Mistral:
  to_empty() leaves inv_freq as uninitialized zeros.
  model.to(bfloat16) locks in those zeros.
  Result: all attention positions get identical encodings → garbage output.

Fix in chat.py: _reinit_rope_buffers() recomputes inv_freq from config hyperparameters
after to_empty() + model.to(dtype).

These tests verify:
  1. inv_freq is non-zero after load (the fix works)
  2. inv_freq values match the expected rope_theta formula
  3. inv_freq stays float32 (never gets quantized to bfloat16)
  4. Regression test: cos similarity at layer_00 must be > 0.999

    pytest proofofconcept/tests/test_rope_initialization.py -v \
        --model ~/workspace/model/Qwen3-1.7B
"""

import sys
import math
from pathlib import Path
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
        p = Path("~/workspace/model/Qwen3-1.7B").expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    cache = p / "codebook" / "tensors"
    if not cache.exists() or not list(cache.glob("*.npz")):
        pytest.skip(f"No codebook cache at {cache}")
    return p


@pytest.fixture(scope="session")
def loaded_model(model_path):
    from chat import CompressedChatModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cm = CompressedChatModel(str(model_path), device='cpu', compression_mode='lossless')
    result = cm.load()
    if result is None:
        pytest.skip("Model failed to load")
    cm.model.eval()
    return cm, tok


@pytest.fixture(scope="session")
def rope_theta(model_path):
    """Get the expected rope_theta from config."""
    import json
    cfg = json.load(open(model_path / "config.json"))
    theta = cfg.get("rope_theta") or cfg.get("rope_parameters", {}).get("rope_theta") or 10000.0
    return float(theta)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRoPEBuffers:
    def test_inv_freq_exists(self, loaded_model):
        cm, _ = loaded_model
        inv_freq_modules = [
            (n, m) for n, m in cm.model.named_modules()
            if hasattr(m, 'inv_freq') and m.inv_freq is not None
        ]
        if not inv_freq_modules:
            pytest.skip("No inv_freq buffers found (model may not use RoPE)")
        print(f"\n  Found {len(inv_freq_modules)} inv_freq buffers")

    def test_inv_freq_not_all_zero(self, loaded_model):
        """The core regression test: inv_freq must not be zeros after load."""
        cm, _ = loaded_model
        bad = []
        for name, mod in cm.model.named_modules():
            if not hasattr(mod, 'inv_freq') or mod.inv_freq is None:
                continue
            if mod.inv_freq.numel() == 0:
                continue
            max_abs = mod.inv_freq.float().abs().max().item()
            if max_abs < 1e-15:
                bad.append((name, max_abs))
        if bad:
            for n, v in bad[:5]:
                print(f"\n  ❌ {n}: inv_freq max_abs={v:.2e} (all zero — RoPE broken)")
        assert not bad, \
            f"{len(bad)} inv_freq buffer(s) are all-zero — _reinit_rope_buffers() failed or wasn't called"

    def test_inv_freq_is_float32(self, loaded_model):
        """inv_freq must stay float32 (not bfloat16) to avoid precision loss."""
        cm, _ = loaded_model
        bad = []
        for name, mod in cm.model.named_modules():
            if not hasattr(mod, 'inv_freq') or mod.inv_freq is None:
                continue
            if mod.inv_freq.numel() == 0:
                continue
            dt = mod.inv_freq.dtype
            if dt != torch.float32:
                bad.append((name, dt))
        if bad:
            for n, dt in bad[:5]:
                print(f"\n  ❌ {n}: inv_freq dtype={dt} (should be float32)")
        assert not bad, \
            f"{len(bad)} inv_freq buffer(s) have wrong dtype (must be float32)"

    def test_inv_freq_values_match_rope_theta_formula(self, loaded_model, rope_theta):
        """
        inv_freq values must match 1/(rope_theta^(2i/d)) for i=0..d/2-1.

        This verifies that _reinit_rope_buffers() used the correct formula and
        the correct theta value from the model config.
        """
        cm, _ = loaded_model
        checked = 0
        for name, mod in cm.model.named_modules():
            if not hasattr(mod, 'inv_freq') or mod.inv_freq is None:
                continue
            buf = mod.inv_freq.float()
            if buf.numel() == 0:
                continue
            half_dim = buf.numel()
            expected = 1.0 / (rope_theta ** (
                torch.arange(0, half_dim * 2, 2, dtype=torch.float32) / (half_dim * 2)
            ))
            max_diff = (buf - expected).abs().max().item()
            rel_diff = max_diff / (expected.abs().max().item() + 1e-30)
            print(f"\n  {name}: half_dim={half_dim}, max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
            assert rel_diff < 1e-4, \
                f"inv_freq values in '{name}' don't match rope_theta={rope_theta:.0f} formula: rel_diff={rel_diff:.2e}"
            checked += 1
            break  # Check first module only — they all share the same rotary emb
        if checked == 0:
            pytest.skip("No inv_freq buffers to check")

    def test_inv_freq_monotonically_decreasing(self, loaded_model):
        """inv_freq values must be monotonically decreasing (1.0 down to near 0).

        This is a sanity check on the formula output:
        - inv_freq[0] = 1.0 (highest frequency = fine-grained position tracking)
        - inv_freq[-1] ≈ 1/theta (lowest frequency = coarse position tracking)
        """
        cm, _ = loaded_model
        for name, mod in cm.model.named_modules():
            if not hasattr(mod, 'inv_freq') or mod.inv_freq is None:
                continue
            buf = mod.inv_freq.float()
            if buf.numel() < 2:
                continue
            diffs = buf[1:] - buf[:-1]
            increasing = (diffs > 1e-10).sum().item()
            print(f"\n  {name}: inv_freq[0]={buf[0]:.4f} inv_freq[-1]={buf[-1]:.6f}")
            assert increasing == 0, \
                f"inv_freq in '{name}' is not monotonically decreasing: {increasing} increases found"
            break


class TestRoPEEffect:
    """Verify that correctly initialized RoPE actually produces correct inference."""

    @pytest.fixture(scope="class")
    def layer0_outputs(self, loaded_model, model_path):
        """
        Run both uncompressed and compressed forward passes, return layer_00 outputs.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import gc

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        msgs = [{"role": "user", "content": "Write a haiku about compression"}]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(enc, 'input_ids'):
            ids = enc.input_ids
        elif isinstance(enc, dict):
            ids = enc['input_ids']
        else:
            ids = enc

        # Uncompressed
        model_unc = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True
        )
        model_unc.eval()
        unc_out = {}
        def hook_fn(mod, inp, out):
            val = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(val, torch.Tensor):
                unc_out['layer_00'] = val.detach().cpu().float()
        h = list(model_unc.model.children())[-1] if hasattr(model_unc.model, 'layers') else None
        inner = model_unc.model
        layers = getattr(inner, 'layers', None)
        if layers is None:
            pytest.skip("Cannot find layers in model")
        hook = layers[0].register_forward_hook(hook_fn)
        with torch.no_grad():
            model_unc(ids)
        hook.remove()
        del model_unc
        gc.collect()

        # Compressed
        cm, _ = loaded_model
        cmp_out = {}
        def hook_fn2(mod, inp, out):
            val = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(val, torch.Tensor):
                cmp_out['layer_00'] = val.detach().cpu().float()
        inner_cmp = cm.model.model
        layers_cmp = getattr(inner_cmp, 'layers', None)
        if layers_cmp is None:
            pytest.skip("Cannot find layers in compressed model")
        hook2 = layers_cmp[0].register_forward_hook(hook_fn2)
        with torch.no_grad():
            cm.model(ids)
        hook2.remove()

        return unc_out, cmp_out

    def test_layer00_cosine_similarity_with_rope_fix(self, layer0_outputs):
        """
        After the RoPE fix, layer_00 hidden state must have cos > 0.999 vs uncompressed.

        Before the fix: cos ≈ 0.45 (GPS-level broken — attention sees no positions)
        After the fix:  cos > 0.999 (positions correct, lossless weights)
        """
        unc_out, cmp_out = layer0_outputs
        if 'layer_00' not in unc_out or 'layer_00' not in cmp_out:
            pytest.skip("Could not capture layer_00 outputs")
        a = unc_out['layer_00'].flatten()
        b = cmp_out['layer_00'].flatten()
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        print(f"\n  layer_00 cosine similarity: {cos:.6f}")
        assert cos > 0.999, \
            f"layer_00 cos={cos:.6f} — RoPE fix may not be working (expected > 0.999)"
