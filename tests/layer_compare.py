#!/usr/bin/env python3
"""
Layer-by-layer comparison: uncompressed vs GPU-compressed vs CPU-compressed.

Runs sequentially to stay within VRAM budget:
  1. Load uncompressed, run forward, save hook outputs to /tmp, free GPU.
  2. Load compressed (GPU kernel), compare to saved outputs, free GPU.
  3. Load compressed (CPU C kernel, device=cpu), compare to saved outputs.

Both compressed runs use greedy (do_sample=False) for determinism.

Usage:
    python proofofconcept/tests/layer_compare.py ~/workspace/model/Qwen3.5-0.8B/
    python proofofconcept/tests/layer_compare.py ~/workspace/model/Qwen3.5-0.8B/ --cpu-only
"""

import sys
import gc
import argparse
import tempfile
from pathlib import Path

import time
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'proofofconcept' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'proofofconcept'))

from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = "Write a haiku about compression"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()

def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()

def report(label: str, a: torch.Tensor, b: torch.Tensor,
           unc_ms: float = 0, cmp_ms: float = 0) -> float:
    cs = cosine_sim(a, b)
    mad = mean_abs_diff(a, b)
    mx = max_abs_diff(a, b)
    icon = "✅" if cs > 0.999 else ("⚠️ " if cs > 0.99 else "❌")
    timing = f"  unc={unc_ms:5.1f}ms cmp={cmp_ms:5.1f}ms" if unc_ms or cmp_ms else ""
    print(f"  {icon} {label:50s}  cos={cs:.6f}  mean={mad:.2e}  max={mx:.2e}{timing}")
    return cs


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

class LayerCapture:
    """Register forward hooks and store last-token hidden states by layer name."""

    def __init__(self, model: torch.nn.Module):
        self.outputs: dict[str, torch.Tensor] = {}
        self._hooks = []
        self._attach(model)

    def _attach(self, model: torch.nn.Module):
        # Unwrap: Qwen3_5ForConditionalGeneration → model → language_model (if present)
        inner = model
        for attr in ('model', 'language_model'):
            child = getattr(inner, attr, None)
            if child is not None:
                inner = child

        embed = getattr(inner, 'embed_tokens', None)
        if embed is not None:
            self._hooks.append(embed.register_forward_hook(self._hook('embed_tokens')))

        layers = getattr(inner, 'layers', [])
        for i, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_hook(self._hook(f'layer_{i:02d}')))

        norm = getattr(inner, 'norm', None)
        if norm is not None:
            self._hooks.append(norm.register_forward_hook(self._hook('final_norm')))

        lm_head = getattr(model, 'lm_head', None)
        if lm_head is not None:
            self._hooks.append(lm_head.register_forward_hook(self._hook('lm_head')))

    def _hook(self, name: str):
        self._start_times = {}

        def fn(module, inp, output):
            val = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(val, torch.Tensor):
                # Keep only last token to save memory; move to CPU immediately
                self.outputs[name] = val[:, -1:, :].detach().cpu().float()
        return fn

    def _timing_pre_hook(self, name: str):
        def fn(module, inp):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start_times[name] = time.perf_counter()
        return fn

    def _timing_post_hook(self, name: str):
        def fn(module, inp, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self._start_times.get(name, time.perf_counter())) * 1000
            self.timings[name] = elapsed
        return fn

    def attach_timings(self, model: torch.nn.Module):
        """Attach separate timing hooks (call after _attach)."""
        self.timings: dict[str, float] = {}
        self._start_times = {}
        inner = model
        for attr in ('model', 'language_model'):
            child = getattr(inner, attr, None)
            if child is not None:
                inner = child
        layers = getattr(inner, 'layers', [])
        for i, layer in enumerate(layers):
            name = f'layer_{i:02d}'
            self._hooks.append(layer.register_forward_pre_hook(self._timing_pre_hook(name)))
            self._hooks.append(layer.register_forward_hook(self._timing_post_hook(name)))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Tokenise
# ---------------------------------------------------------------------------

def get_input_ids(tokenizer, model_path: Path) -> torch.Tensor:
    messages = [{"role": "user", "content": PROMPT}]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(tokenized, 'input_ids'):
        return tokenized.input_ids
    elif isinstance(tokenized, dict):
        return tokenized['input_ids']
    return tokenized


# ---------------------------------------------------------------------------
# Phase 1 – uncompressed
# ---------------------------------------------------------------------------

def run_uncompressed(model_path: Path, save_dir: Path):
    print("\n=== Phase 1: UNCOMPRESSED forward pass ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    input_ids = get_input_ids(tokenizer, model_path).to(DEVICE)
    print(f"  Prompt: '{PROMPT}'  ({input_ids.shape[1]} tokens)")

    cap = LayerCapture(model)
    cap.attach_timings(model)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    cap.remove()

    # Greedy first 5 tokens
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False,
                             temperature=1.0, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    unc_tokens = out[0, input_ids.shape[1]:].tolist()
    unc_text = tokenizer.decode(unc_tokens, skip_special_tokens=True)
    print(f"  Greedy 5 tokens: {unc_tokens}  → '{unc_text}'")

    # Save outputs and timings
    save_dir.mkdir(exist_ok=True)
    for name, tensor in cap.outputs.items():
        np.save(save_dir / f"{name}.npy", tensor.numpy())
    np.save(save_dir / "_greedy_tokens.npy", np.array(unc_tokens))
    timing_arr = np.array([[k, v] for k, v in cap.timings.items()], dtype=object)
    np.save(save_dir / "_timings.npy", timing_arr, allow_pickle=True)
    print(f"  Saved {len(cap.outputs)} layer outputs + {len(cap.timings)} timings to {save_dir}")

    # Free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU after free: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, set(cap.outputs.keys())


# ---------------------------------------------------------------------------
# Phase 2 – compressed (GPU kernel or CPU C kernel)
# ---------------------------------------------------------------------------

def run_compressed(model_path: Path, save_dir: Path, layer_names: set, device: str = None):
    dev = device or DEVICE
    label = f"COMPRESSED ({dev.upper()} {'GPU kernel' if dev != 'cpu' else 'C kernel'})"
    print(f"\n=== Phase 2: {label} ===")
    from chat import CompressedChatModel

    cm = CompressedChatModel(model_path, device=dev, compression_mode='lossless')
    cm.load()
    model = cm.model
    tokenizer = cm.tokenizer
    if dev != 'cpu':
        print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    input_ids = get_input_ids(tokenizer, model_path).to(dev)

    cap = LayerCapture(model)
    cap.attach_timings(model)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    cap.remove()

    # Greedy first 5 tokens
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False,
                             temperature=1.0, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    cmp_tokens = out[0, input_ids.shape[1]:].tolist()
    cmp_text = tokenizer.decode(cmp_tokens, skip_special_tokens=True)
    print(f"  Greedy 5 tokens: {cmp_tokens}  → '{cmp_text}'")

    # Free GPU memory if used
    del model, cm
    gc.collect()
    if dev != 'cpu':
        torch.cuda.empty_cache()

    return cap.outputs, cap.timings, cmp_tokens, cmp_text


# ---------------------------------------------------------------------------
# Compare helper
# ---------------------------------------------------------------------------

def compare(save_dir: Path, cmp_outputs: dict, cmp_timings: dict,
            cmp_tokens: list, cmp_text: str, tokenizer, label: str = "Compressed"):
    print(f"\n=== Layer comparison: Uncompressed vs {label} (last-token hidden state) ===\n")

    unc_tokens = np.load(save_dir / "_greedy_tokens.npy").tolist()
    unc_text = tokenizer.decode([int(t) for t in unc_tokens], skip_special_tokens=True)

    unc_timings = {}
    timing_file = save_dir / "_timings.npy"
    if timing_file.exists():
        for row in np.load(timing_file, allow_pickle=True):
            unc_timings[str(row[0])] = float(row[1])

    saved = sorted([f.stem for f in save_dir.glob("*.npy") if not f.stem.startswith('_')],
                   key=lambda k: (0 if k == 'embed_tokens' else
                                  (1 if k.startswith('layer_') else
                                   (2 if k == 'final_norm' else 3)),
                                  k))

    first_fail = None
    results = {}
    for name in saved:
        if name not in cmp_outputs:
            print(f"  ⚠️  {name}: missing from compressed model")
            continue
        unc = torch.from_numpy(np.load(save_dir / f"{name}.npy"))
        cmp = cmp_outputs[name]
        if unc.shape != cmp.shape:
            print(f"  ⚠️  {name}: shape mismatch unc={unc.shape} cmp={cmp.shape}")
            continue
        unc_ms = unc_timings.get(name, 0)
        cmp_ms = cmp_timings.get(name, 0)
        cs = report(name, unc, cmp, unc_ms, cmp_ms)
        results[name] = cs
        if first_fail is None and cs < 0.999:
            first_fail = name

    print()
    if first_fail:
        print(f"🔍 First divergence: {first_fail}  (cos={results[first_fail]:.6f})")
    else:
        print(f"✅ All layers match (cos > 0.999)")

    print(f"\n=== Token comparison: {label} ===")
    print(f"  Uncompressed: {unc_tokens}  → '{unc_text}'")
    print(f"  {label:12s}: {cmp_tokens}  → '{cmp_text}'")
    match = sum(a == b for a, b in zip(unc_tokens, cmp_tokens))
    print(f"  Match: {match}/{len(unc_tokens)}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--save-dir', type=Path,
                        default=Path('/tmp/layer_compare_unc'))
    parser.add_argument('--cpu-only', action='store_true',
                        help='Skip GPU phase, only run CPU C kernel comparison')
    parser.add_argument('--no-cpu', action='store_true',
                        help='Skip CPU phase (faster if GPU kernel is the focus)')
    args = parser.parse_args()

    model_path = args.model_path.expanduser()

    tokenizer, layer_names = run_uncompressed(model_path, args.save_dir)

    if not args.cpu_only and DEVICE != 'cpu':
        gpu_outputs, gpu_timings, gpu_tokens, gpu_text = run_compressed(
            model_path, args.save_dir, layer_names, device=DEVICE)
        compare(args.save_dir, gpu_outputs, gpu_timings, gpu_tokens, gpu_text,
                tokenizer, label="GPU kernel")

    if not args.no_cpu:
        cpu_outputs, cpu_timings, cpu_tokens, cpu_text = run_compressed(
            model_path, args.save_dir, layer_names, device='cpu')
        compare(args.save_dir, cpu_outputs, cpu_timings, cpu_tokens, cpu_text,
                tokenizer, label="CPU C kernel")


if __name__ == '__main__':
    main()
