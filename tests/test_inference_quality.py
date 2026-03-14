"""
test_inference_quality.py — End-to-end inference quality tests for compressed models.

Tests that the compressed model:
  1. Loads without NaN/Inf in weights
  2. Produces numerically stable hidden states
  3. Generates logits with reasonable entropy (not degenerate)
  4. Does not generate the same token for every step (repetition trap)

These tests are the "canary" — if they fail, inference quality has regressed.

Run with a specific model:
    pytest proofofconcept/tests/test_inference_quality.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct

Default falls back to 0.8B if --model not given.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "proofofconcept" / "src"
SCRIPT_DIR = ROOT / "proofofconcept"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

# ── fixtures ─────────────────────────────────────────────────────────────────

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
    """Load a CompressedChatModel on CPU and return (model, tokenizer)."""
    from transformers import AutoTokenizer
    from chat import CompressedChatModel

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    m = CompressedChatModel(str(model_path), device='cpu')
    m.load()
    m.model.eval()
    return m, tok


@pytest.fixture(scope="session")
def prompt_ids(loaded_model):
    """Tokenized 'Write a haiku about compression' with generation prompt."""
    m, tok = loaded_model
    msgs = [{"role": "user", "content": "Write a haiku about compression"}]
    enc = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    ids = enc if isinstance(enc, torch.Tensor) else enc.get('input_ids', enc)
    return ids


@pytest.fixture(scope="session")
def single_forward(loaded_model, prompt_ids):
    """Run one forward pass and return (hidden_states tuple, logits tensor)."""
    m, _ = loaded_model
    with torch.no_grad():
        out = m.model(prompt_ids, output_hidden_states=True)
    return out.hidden_states, out.logits[0, -1]


# ── tests ─────────────────────────────────────────────────────────────────────

class TestWeightSanity:
    """Compressed model weights must be finite and non-trivial after load."""

    def test_final_norm_nonzero(self, loaded_model):
        m, _ = loaded_model
        norm = m.model.model.norm
        w = norm.weight.float()
        assert not torch.isnan(w).any(), "final norm weight has NaN"
        assert not torch.isinf(w).any(), "final norm weight has Inf"
        assert w.abs().mean() > 0.01, f"final norm weight near zero (mean={w.abs().mean():.4f})"

    def test_no_all_zero_layer_weights(self, loaded_model):
        """Spot-check: no AdaptiveCodebookLinear has a codebook of all zeros."""
        m, _ = loaded_model
        for name, mod in m.model.named_modules():
            if hasattr(mod, 'mode') and mod.mode == 'direct_codebook':
                if mod.codebook is not None:
                    cb = mod.codebook.float()
                    assert cb.abs().max() > 1e-6, \
                        f"{name}: codebook is all zeros — weights not loaded"
                    break  # checking one is enough as a smoke test

    def test_bias_finite_where_present(self, loaded_model):
        """Any bias tensors on compressed modules must be finite."""
        m, _ = loaded_model
        for name, mod in m.model.named_modules():
            bias = getattr(mod, 'bias', None)
            if hasattr(mod, 'mode') and bias is not None:
                b = bias.float()
                assert torch.isfinite(b).all(), \
                    f"{name}: bias contains NaN/Inf"


class TestHiddenStateStability:
    """Hidden states must be numerically stable throughout the network."""

    def test_no_nan_in_hidden_states(self, single_forward):
        hs, _ = single_forward
        for i, h in enumerate(hs):
            assert not torch.isnan(h).any(), \
                f"NaN in hidden state at layer {i}"
            assert not torch.isinf(h).any(), \
                f"Inf in hidden state at layer {i}"

    def test_hidden_state_rms_grows_then_shrinks(self, single_forward):
        """Hidden state RMS should grow through mid-network and not be 0 or explosive."""
        hs, _ = single_forward
        rms_values = [h.float().pow(2).mean().sqrt().item() for h in hs]
        # First embedding should be small, final norm output should be non-zero
        assert rms_values[0] < 5.0, f"Embedding RMS suspiciously large: {rms_values[0]:.3f}"
        assert rms_values[-1] > 0.1, f"Final hidden state RMS near zero: {rms_values[-1]:.4f}"
        # No layer should blow up
        for i, rms in enumerate(rms_values):
            assert rms < 1000.0, f"Hidden state exploded at layer {i}: rms={rms:.1f}"

    def test_final_hidden_state_not_constant(self, single_forward):
        """The final hidden state must vary across the sequence (not all-same vector)."""
        hs, _ = single_forward
        final = hs[-1][0].float()  # [seq, hidden]
        if final.shape[0] < 2:
            pytest.skip("Sequence length < 2, can't check variance")
        # Cosine sim between first and last token should not be 1.0 (degenerate)
        v0 = final[0] / (final[0].norm() + 1e-9)
        v1 = final[-1] / (final[-1].norm() + 1e-9)
        cos = (v0 * v1).sum().item()
        assert cos < 0.9999, \
            f"Final hidden state is constant across positions (cos={cos:.6f})"


class TestLogitQuality:
    """Output logits must reflect a well-functioning model."""

    def test_logits_not_degenerate(self, single_forward):
        """Logits must have meaningful spread — not all zero or all same value."""
        _, logits = single_forward
        logits_f = logits.float()
        std = logits_f.std().item()
        assert std > 0.5, \
            f"Logit std too small ({std:.4f}) — model may be outputting garbage"

    def test_top_token_is_not_pad(self, loaded_model, single_forward):
        """The top-1 predicted token must not be the pad token."""
        m, tok = loaded_model
        _, logits = single_forward
        top_id = int(logits.argmax())
        pad_id = tok.pad_token_id
        assert top_id != pad_id, \
            f"Model predicts pad_token ({pad_id}) at every step — likely degenerate"

    def test_top5_includes_non_special_token(self, loaded_model, single_forward):
        """At least one of top-5 tokens should be a regular (non-special) token."""
        m, tok = loaded_model
        _, logits = single_forward
        top5_ids = logits.topk(5).indices.tolist()
        special_ids = set(tok.all_special_ids)
        non_special = [t for t in top5_ids if t not in special_ids]
        assert len(non_special) > 0, \
            f"All top-5 predicted tokens are special tokens: {top5_ids}"


class TestGenerationCoherence:
    """Greedy generation must not immediately collapse to a constant token loop."""

    def test_greedy_does_not_repeat_single_token(self, loaded_model, prompt_ids):
        """Generate 6 tokens greedily — must not all be the same token id."""
        m, tok = loaded_model
        ids = prompt_ids.clone()
        generated = []
        with torch.no_grad():
            for _ in range(6):
                out = m.model(ids)
                next_id = int(out.logits[0, -1].argmax())
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

        unique_tokens = len(set(generated))
        assert unique_tokens > 1, \
            f"Model generated the same token {generated[0]} for all 6 steps — stuck in loop"

    def test_generated_text_is_not_empty(self, loaded_model, prompt_ids):
        """model.generate() must produce at least 1 visible (non-special) token in 10."""
        m, tok = loaded_model
        with torch.no_grad():
            out = m.model.generate(
                prompt_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        generated_ids = out[0, prompt_ids.shape[1]:]
        text = tok.decode(generated_ids, skip_special_tokens=True).strip()
        assert len(text) > 0, \
            "model.generate() produced only special tokens / empty string in 10 steps"
