"""
test_layer_compare.py — Integration test: layer-by-layer cosine similarity
between uncompressed and lossless-compressed Qwen3.5-0.8B.

Requires:
  - Model at ~/workspace/model/Qwen3.5-0.8B/
  - Pre-built lossless codebook at .../codebook/tensors/

Run with:
    pytest proofofconcept/tests/test_layer_compare.py -v --run-slow

Skip if model is not present (marked integration + slow).
"""

import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

DEFAULT_MODEL = Path("~/workspace/model/Qwen3.5-0.8B").expanduser()
PROMPT = "Write a haiku about compression"
GREEDY_TOK = 5          # how many greedy tokens to compare
COS_THRESHOLD = 0.999   # minimum acceptable cosine similarity per layer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _greedy_tokens(logits: torch.Tensor, k: int) -> list:
    return logits[0, -1].topk(k).indices.tolist()


def _collect_layer_outputs(model, input_ids) -> dict:
    """Run a single forward pass and collect last-token hidden states per layer."""
    hooks = []
    outputs = {}

    def make_hook(name):
        def hook(module, inp, out):
            # For layers returning tuples, take first element
            h = out[0] if isinstance(out, tuple) else out
            outputs[name] = h[:, -1, :].detach().cpu().float()
        return hook

    # Register hooks on each transformer block
    def _register(parent, prefix=""):
        for name, child in parent.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if "embed_tokens" in full or "model" in full:
                child.register_forward_hook(make_hook(full))
                h = child.register_forward_hook(make_hook(full))
                hooks.append(h)
            _register(child, full)

    # Simpler: hook at the model's transformer layer list
    lang_model = getattr(model, "model", model)
    embed = getattr(lang_model, "embed_tokens", None)
    if embed is not None:
        h = embed.register_forward_hook(make_hook("embed_tokens"))
        hooks.append(h)

    layers = getattr(lang_model, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(make_hook(f"layer_{i:02d}"))
            hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    for h in hooks:
        h.remove()
    return outputs, logits


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_path():
    p = DEFAULT_MODEL
    if not p.exists() or not (p / "codebook" / "tensors").exists():
        pytest.skip(f"Model or codebook not found at {p}")
    npz_count = len(list((p / "codebook" / "tensors").glob("*.npz")))
    if npz_count == 0:
        pytest.skip(f"Codebook tensors directory is empty — run compress.py first")
    return p


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestLayerCompare:

    def test_greedy_tokens_match(self, model_path):
        """Compressed model must predict the same greedy tokens as uncompressed."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from chat import CompressedChatModel

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        input_ids = tok(PROMPT, return_tensors="pt").input_ids

        # --- uncompressed ---
        unc_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True
        ).eval()
        with torch.no_grad():
            unc_logits = unc_model(input_ids=input_ids.to(next(unc_model.parameters()).device)).logits
        unc_tokens = _greedy_tokens(unc_logits.cpu(), GREEDY_TOK)
        del unc_model
        gc.collect()
        torch.cuda.empty_cache()

        # --- compressed ---
        cm = CompressedChatModel(model_path, compression_mode="lossless")
        assert cm.load() is not None, "Compressed model failed to load"
        with torch.no_grad():
            cmp_logits = cm.model(input_ids=input_ids.to(cm.device)).logits
        cmp_tokens = _greedy_tokens(cmp_logits.cpu(), GREEDY_TOK)
        del cm
        gc.collect()
        torch.cuda.empty_cache()

        assert unc_tokens == cmp_tokens, (
            f"Greedy token mismatch.\n  Uncompressed: {unc_tokens}\n  Compressed:   {cmp_tokens}"
        )

    def test_all_layers_cosine_similarity(self, model_path):
        """Every transformer layer output must have cosine similarity > COS_THRESHOLD."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from chat import CompressedChatModel

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        input_ids = tok(PROMPT, return_tensors="pt").input_ids

        # --- uncompressed ---
        unc_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True
        ).eval()
        unc_outputs, _ = _collect_layer_outputs(unc_model, input_ids.to(next(unc_model.parameters()).device))
        del unc_model
        gc.collect()
        torch.cuda.empty_cache()

        # --- compressed ---
        cm = CompressedChatModel(model_path, compression_mode="lossless")
        assert cm.load() is not None
        cmp_outputs, _ = _collect_layer_outputs(cm.model, input_ids.to(cm.device))
        del cm
        gc.collect()
        torch.cuda.empty_cache()

        failures = []
        for name in unc_outputs:
            if name not in cmp_outputs:
                continue
            cos = _cosine_sim(unc_outputs[name], cmp_outputs[name])
            if cos < COS_THRESHOLD:
                failures.append(f"  {name}: cos={cos:.6f} < {COS_THRESHOLD}")

        assert not failures, "Layer cosine similarity failures:\n" + "\n".join(failures)

    def test_generate_greedy_coherent(self, model_path):
        """Compressed model must produce non-empty coherent text with greedy decode."""
        from chat import CompressedChatModel

        cm = CompressedChatModel(model_path, compression_mode="lossless")
        assert cm.load() is not None

        text = cm.generate(
            [{"role": "user", "content": PROMPT}],
            max_tokens=30,
            temperature=0.0,
        )
        assert isinstance(text, str)
        assert len(text.strip()) > 0, "Empty response from compressed model"
        # Basic coherence check: should not be pure repetition of one codepoint
        chars = set(text.strip())
        assert len(chars) > 3, f"Response looks repetitive (only {len(chars)} unique chars): {text[:100]!r}"
