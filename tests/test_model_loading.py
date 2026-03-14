"""
Regression tests for model loading bugs.

test_conv1d_weight_not_nan:
    Regression for the bug introduced in commit 8de00d4 where _load_exact_weights
    skipped ALL direct_codebook tensors, including nn.Conv1d weights (e.g.
    linear_attn.conv1d.weight).  These were left as NaN from model.to_empty(),
    causing NaN to propagate from layer_00 onward.

    The fix: only skip direct_codebook tensors whose parent module is nn.Linear
    or nn.Embedding (those will be replaced by _replace_modules_recursive).
    All other module types (Conv1d etc.) are decompressed and loaded.
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = Path(os.environ.get(
    'COMPRESS_MODEL_PATH',
    os.path.expanduser('~/workspace/model/Qwen3.5-0.8B')
))

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH / 'codebook').exists(),
    reason=f"Compressed model cache not found at {MODEL_PATH}/codebook"
)


def _load_compressed_model(device='cpu'):
    """Load via chat.CompressedChatModel (same path as inference)."""
    import warnings
    warnings.filterwarnings('ignore')
    from chat import CompressedChatModel
    cm = CompressedChatModel(MODEL_PATH, device=device, compression_mode='lossless')
    cm.load()
    return cm


class TestModelLoadingNoNaN:
    """No parameters or buffers should be NaN/uninit after model load."""

    def test_no_nan_parameters(self):
        """All parameters must be finite after compressed load (no to_empty() leftovers)."""
        cm = _load_compressed_model(device='cpu')
        bad = []
        for name, p in cm.model.named_parameters():
            if p.numel() > 0 and p.isnan().any():
                bad.append(f"{name} ({p.shape})")
        assert not bad, (
            f"NaN parameters after load — likely direct_codebook weights left "
            f"uninitialized (Conv1d or similar):\n  " + "\n  ".join(bad[:10])
        )

    def test_conv1d_weights_loaded(self):
        """Conv1d weights in SSM layers must not be NaN (regression for 8de00d4 bug)."""
        cm = _load_compressed_model(device='cpu')
        conv_bad = []
        for name, mod in cm.model.named_modules():
            if isinstance(mod, torch.nn.Conv1d):
                if mod.weight.isnan().any():
                    conv_bad.append(name)
        assert not conv_bad, (
            f"Conv1d weights are NaN — _load_exact_weights is skipping them instead "
            f"of decompressing:\n  " + "\n  ".join(conv_bad)
        )

    def test_layer0_forward_not_nan(self):
        """A single forward pass through layer 0 must not produce NaN."""
        cm = _load_compressed_model(device='cpu')
        from transformers import AutoTokenizer
        import warnings
        warnings.filterwarnings('ignore')

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        ids = tokenizer.encode("Hi", return_tensors='pt')

        with torch.no_grad():
            out = cm.model(input_ids=ids)

        logits = out.logits
        assert not logits.isnan().any(), (
            "Model output logits contain NaN — some layer weight was not loaded correctly"
        )
        assert not logits.isinf().any(), "Model output logits contain Inf"

    def test_greedy_tokens_match_uncompressed(self):
        """Greedy first 5 tokens must match the uncompressed model."""
        import warnings
        warnings.filterwarnings('ignore')
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "Write a haiku about compression"
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(ids, 'input_ids'):
            ids = ids.input_ids

        # Uncompressed
        unc_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True
        )
        unc_model.eval()
        with torch.no_grad():
            unc_out = unc_model.generate(
                input_ids=ids, max_new_tokens=5, do_sample=False,
                temperature=1.0, pad_token_id=tokenizer.pad_token_id,
            )
        unc_tokens = unc_out[0, ids.shape[1]:].tolist()
        del unc_model

        # Compressed
        cm = _load_compressed_model(device='cpu')
        with torch.no_grad():
            cmp_out = cm.model.generate(
                input_ids=ids, max_new_tokens=5, do_sample=False,
                temperature=1.0, pad_token_id=tokenizer.pad_token_id,
            )
        cmp_tokens = cmp_out[0, ids.shape[1]:].tolist()

        matches = sum(a == b for a, b in zip(unc_tokens, cmp_tokens))
        assert matches >= 4, (
            f"Greedy tokens diverge too much: "
            f"unc={unc_tokens} cmp={cmp_tokens} ({matches}/5 match)\n"
            f"This may indicate a weight loading regression."
        )
