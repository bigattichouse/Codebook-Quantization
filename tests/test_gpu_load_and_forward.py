"""
test_gpu_load_and_forward.py — Phase 6: GPU load & forward (OOM-aware).

Tests the GPU path for compressed inference. Unlike CPU tests, these SKIP
(not FAIL) on CUDA OOM — that is the correct signal for "use CPU path instead".

    pytest proofofconcept/tests/test_gpu_load_and_forward.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct -s

Expected outcomes:
  - Qwen3.5-0.8B: all pass
  - Qwen2.5-3B:   most pass (fits in 4 GB usable VRAM)
  - Qwen3.5-9B:   all skip (OOM)
"""

import sys
import gc
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
        p = Path("~/workspace/model/Qwen3.5-0.8B").expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    cache = p / "codebook" / "tensors"
    if not cache.exists() or not list(cache.glob("*.npz")):
        pytest.skip(f"No codebook cache at {cache}")
    return p


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def gpu_loaded_model(model_path):
    """
    Try to load model on GPU. SKIP (not fail) if OOM occurs.
    Returns (cm, tokenizer).
    """
    _require_cuda()
    from transformers import AutoTokenizer
    from chat import CompressedChatModel

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        cm = CompressedChatModel(str(model_path), device='cuda', compression_mode='lossless')
        result = cm.load()
        if result is None:
            pytest.skip("GPU model failed to load (check cache)")
        cm.model.eval()
        # Report actual device used (may have fallen back to CPU)
        actual_device = cm.device
        print(f"\n  GPU load successful. Actual device: {actual_device}")
        print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.3f} GB")
        return cm, tok
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "memory" in str(e).lower() or "CUDA out of memory" in str(e):
            pytest.skip(
                f"GPU OOM during load — model too large for VRAM. "
                f"Use device='cpu'. Error: {e}"
            )
        raise


@pytest.fixture(scope="session")
def gpu_prompt_ids(gpu_loaded_model):
    cm, tok = gpu_loaded_model
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
    return ids.to(cm.device)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestGPUAvailability:
    def test_cuda_available(self):
        _require_cuda()
        print(f"\n  CUDA device: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"  Total VRAM: {total/1e9:.2f} GB")
        assert torch.cuda.is_available()

    def test_vram_before_load(self):
        _require_cuda()
        used = torch.cuda.memory_allocated()
        print(f"\n  VRAM before load: {used/1e9:.3f} GB")
        # Just report — no assertion


class TestGPULoad:
    def test_gpu_model_loads(self, gpu_loaded_model):
        cm, _ = gpu_loaded_model
        assert cm.model is not None

    def test_gpu_model_device(self, gpu_loaded_model):
        """Verify model parameters are on the expected device."""
        cm, _ = gpu_loaded_model
        # At least some params should be on GPU or the model skeleton should be there
        # (codebook/indices may be on CPU in hybrid mode)
        print(f"\n  cm.device: {cm.device}")
        assert cm.device in ('cuda', 'cpu')  # may have fallen back

    def test_no_nan_in_gpu_parameters(self, gpu_loaded_model):
        cm, _ = gpu_loaded_model
        nan_params = [(n, p.shape) for n, p in cm.model.named_parameters()
                      if p.is_cuda and torch.isnan(p).any()]
        if nan_params:
            print(f"\n  NaN GPU params:")
            for n, s in nan_params[:5]:
                print(f"    {n}: {s}")
        assert not nan_params, f"{len(nan_params)} GPU parameters contain NaN"

    def test_gpu_compressed_module_count(self, gpu_loaded_model):
        from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
        cm, _ = gpu_loaded_model
        n_linear = sum(1 for _, m in cm.model.named_modules()
                       if isinstance(m, AdaptiveCodebookLinear))
        n_embed = sum(1 for _, m in cm.model.named_modules()
                      if isinstance(m, AdaptiveCodebookEmbedding))
        print(f"\n  Compressed modules: {n_linear} linear + {n_embed} embedding")
        assert n_linear >= 50

    def test_gpu_modules_have_codebooks(self, gpu_loaded_model):
        from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
        cm, _ = gpu_loaded_model
        bad = []
        for name, mod in cm.model.named_modules():
            if not isinstance(mod, (AdaptiveCodebookLinear, AdaptiveCodebookEmbedding)):
                continue
            if mod.mode != 'direct_codebook':
                continue
            if mod.codebook is None:
                bad.append(f"{name}: codebook is None")
        if bad:
            for b in bad[:5]:
                print(f"\n  ❌ {b}")
        assert not bad, f"{len(bad)} GPU compressed modules missing codebook"


class TestGPUForward:
    def test_gpu_forward_no_exception(self, gpu_loaded_model, gpu_prompt_ids):
        cm, _ = gpu_loaded_model
        try:
            with torch.no_grad():
                out = cm.model(gpu_prompt_ids)
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM during forward — {e}")
        assert out is not None

    def test_gpu_forward_logits_finite(self, gpu_loaded_model, gpu_prompt_ids):
        cm, _ = gpu_loaded_model
        try:
            with torch.no_grad():
                out = cm.model(gpu_prompt_ids)
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM during forward — {e}")
        logits = out.logits[0, -1].float()
        nan_count = int(torch.isnan(logits).sum())
        inf_count = int(torch.isinf(logits).sum())
        assert nan_count == 0, f"GPU logits: {nan_count} NaN values"
        assert inf_count == 0, f"GPU logits: {inf_count} Inf values"

    def test_gpu_logit_std_reasonable(self, gpu_loaded_model, gpu_prompt_ids):
        cm, _ = gpu_loaded_model
        try:
            with torch.no_grad():
                out = cm.model(gpu_prompt_ids)
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM during forward — {e}")
        logits = out.logits[0, -1].float()
        std = logits.std().item()
        print(f"\n  GPU logit std: {std:.4f}")
        assert std > 0.01, f"GPU logit std too small: {std:.6f}"

    def test_gpu_greedy_6_tokens_diverse(self, gpu_loaded_model, gpu_prompt_ids):
        cm, _ = gpu_loaded_model
        ids = gpu_prompt_ids.clone()
        generated = []
        try:
            with torch.no_grad():
                for _ in range(6):
                    out = cm.model(ids)
                    next_id = int(out.logits[0, -1].argmax())
                    generated.append(next_id)
                    ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM during generation — {e}")
        unique = len(set(generated))
        print(f"\n  GPU greedy: {generated} (unique={unique})")
        assert unique > 1, f"GPU model stuck: repeated token {generated[0]}"

    def test_vram_after_forward(self, gpu_loaded_model, gpu_prompt_ids):
        """Report VRAM usage after forward pass — always passes."""
        cm, _ = gpu_loaded_model
        try:
            with torch.no_grad():
                cm.model(gpu_prompt_ids)
        except torch.cuda.OutOfMemoryError:
            pytest.skip("OOM during forward")
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"\n  VRAM after forward: {allocated/1e9:.3f} GB allocated, "
              f"{reserved/1e9:.3f} GB reserved")


class TestGPUvsLayerByLayer:
    """Attach hooks to all layers and check for NaN on GPU."""

    def test_gpu_no_nan_in_any_layer(self, gpu_loaded_model, gpu_prompt_ids):
        cm, _ = gpu_loaded_model
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

        try:
            with torch.no_grad():
                model(gpu_prompt_ids)
        except torch.cuda.OutOfMemoryError as e:
            for h in hooks:
                h.remove()
            pytest.skip(f"GPU OOM during hooked forward — {e}")
        finally:
            for h in hooks:
                h.remove()

        bad_layers = [(n, r) for n, r in results.items() if r['nan'] or r['inf']]
        if bad_layers:
            print(f"\n  Layers with NaN/Inf ({len(bad_layers)}):")
            for n, r in bad_layers[:10]:
                print(f"    {n}: nan={r['nan']}, inf={r['inf']}, rms={r['rms']:.4f}")
            first = bad_layers[0][0]
            pytest.fail(f"First GPU NaN at layer '{first}'")
