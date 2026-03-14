#!/usr/bin/env python3
"""
Three-way inference benchmark: Uncompressed GPU / Compressed GPU / Compressed CPU.

Reports for each mode:
  - CPU RAM delta (MB): RSS before load → after load → peak during generation
  - VRAM delta (MB): GPU memory before → after load → peak during generation
  - Tokens/sec (generation phase only, excludes load time)
  - Generated text

Usage:
    python benchmark.py ~/workspace/model/Qwen3.5-0.8B
    python benchmark.py ~/workspace/model/Qwen3.5-0.8B --prompt "Write a haiku about compression" --tokens 30
    python benchmark.py ~/workspace/model/Qwen3.5-0.8B --mode gpu        # skip CPU (slow)
    python benchmark.py ~/workspace/model/Qwen3.5-0.8B --mode cpu        # skip uncompressed
"""

import argparse
import gc
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import psutil
import torch

MODEL_PATH = Path(os.environ.get('COMPRESS_MODEL_PATH',
                  os.path.expanduser('~/workspace/model/Qwen3.5-0.8B')))


def _rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _vram_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 / 1024


def _vram_peak_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def _reset_vram_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tokenize(model_path, prompt):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_tensors="pt")
    if hasattr(ids, 'input_ids'):
        ids = ids.input_ids
    return ids, tok


def run_uncompressed(model_path, prompt, max_tokens):
    """Uncompressed model on GPU with device_map='cuda'."""
    from transformers import AutoModelForCausalLM
    print("\n" + "=" * 60)
    print("MODE: UNCOMPRESSED GPU")
    print("=" * 60)

    _gc()
    rss_base = _rss_mb()
    _reset_vram_peak()
    vram_base = _vram_mb()

    print(f"  Baseline  — RSS: {rss_base:.0f} MB  VRAM: {vram_base:.0f} MB")
    print("  Loading model...", flush=True)
    t_load = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            _gc()
            print(f"  *** CUDA OOM — model too large for {_vram_mb():.0f} MB available VRAM ***")
            print(f"  (This is expected — compression is required to run this model)")
            return dict(mode="Uncompressed GPU", oom=True,
                        rss_base=rss_base, rss_load=0, rss_gen_delta=0,
                        vram_load=0, vram_peak_gen=0, tps=0, load_s=0, text="OOM")
        raise
    model.eval()
    ids, tok = _tokenize(model_path, prompt)
    ids = ids.to(device)

    t_load = time.time() - t_load
    _gc()
    rss_after_load = _rss_mb()
    vram_after_load = _vram_mb()
    print(f"  After load — RSS: {rss_after_load:.0f} MB (+{rss_after_load - rss_base:.0f})  "
          f"VRAM: {vram_after_load:.0f} MB (+{vram_after_load - vram_base:.0f})  "
          f"load: {t_load:.1f}s")

    _reset_vram_peak()
    rss_pre_gen = _rss_mb()

    print(f"  Generating {max_tokens} tokens...", flush=True)
    with torch.no_grad():
        t0 = time.time()
        out = model.generate(
            input_ids=ids, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0, pad_token_id=tok.pad_token_id,
        )
        gen_time = time.time() - t0

    new_tokens = out.shape[1] - ids.shape[1]
    tps = new_tokens / gen_time
    rss_post_gen = _rss_mb()
    vram_peak_gen = _vram_peak_mb()
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    print(f"  Generation — RSS: {rss_post_gen:.0f} MB (Δ{rss_post_gen - rss_pre_gen:+.0f})  "
          f"VRAM peak: {vram_peak_gen:.0f} MB  {tps:.1f} tok/s")
    print(f"  Text: \"{text}\"")

    del model; _gc()
    return dict(mode="Uncompressed GPU",
                rss_base=rss_base, rss_load=rss_after_load - rss_base,
                rss_gen_delta=rss_post_gen - rss_pre_gen,
                vram_load=vram_after_load - vram_base, vram_peak_gen=vram_peak_gen,
                tps=tps, load_s=t_load, text=text, oom=False)


def run_compressed_gpu(model_path, prompt, max_tokens):
    """Compressed model, GPU kernel."""
    from chat import CompressedChatModel
    print("\n" + "=" * 60)
    print("MODE: COMPRESSED GPU")
    print("=" * 60)

    _gc()
    rss_base = _rss_mb()
    _reset_vram_peak()
    vram_base = _vram_mb()

    print(f"  Baseline  — RSS: {rss_base:.0f} MB  VRAM: {vram_base:.0f} MB")
    print("  Loading model...", flush=True)
    t_load = time.time()

    model = CompressedChatModel(model_path, device='cuda', compression_mode='lossless')
    model.load()
    ids, tok = _tokenize(model_path, prompt)
    # Detect actual device after potential GPU→CPU fallback
    try:
        actual_device = next(p for p in model.model.parameters() if p.device.type != 'meta').device
    except StopIteration:
        actual_device = torch.device('cpu')
    ids = ids.to(actual_device)
    actual_mode = "Compressed GPU" if actual_device.type == 'cuda' else "Compressed GPU→CPU fallback"

    t_load = time.time() - t_load
    _gc()
    rss_after_load = _rss_mb()
    vram_after_load = _vram_mb()
    print(f"  After load — RSS: {rss_after_load:.0f} MB (+{rss_after_load - rss_base:.0f})  "
          f"VRAM: {vram_after_load:.0f} MB (+{vram_after_load - vram_base:.0f})  "
          f"load: {t_load:.1f}s")

    _reset_vram_peak()
    rss_pre_gen = _rss_mb()

    print(f"  Generating {max_tokens} tokens...", flush=True)
    model.model.eval()
    with torch.no_grad():
        t0 = time.time()
        out = model.model.generate(
            input_ids=ids, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0, pad_token_id=tok.pad_token_id,
        )
        gen_time = time.time() - t0

    new_tokens = out.shape[1] - ids.shape[1]
    tps = new_tokens / gen_time
    rss_post_gen = _rss_mb()
    vram_peak_gen = _vram_peak_mb()
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    print(f"  Generation — RSS: {rss_post_gen:.0f} MB (Δ{rss_post_gen - rss_pre_gen:+.0f})  "
          f"VRAM peak: {vram_peak_gen:.0f} MB  {tps:.1f} tok/s")
    print(f"  Text: \"{text}\"")

    del model; _gc()
    return dict(mode=actual_mode,
                rss_base=rss_base, rss_load=rss_after_load - rss_base,
                rss_gen_delta=rss_post_gen - rss_pre_gen,
                vram_load=vram_after_load - vram_base, vram_peak_gen=vram_peak_gen,
                tps=tps, load_s=t_load, text=text)


def run_compressed_cpu(model_path, prompt, max_tokens):
    """Compressed model, CPU C kernel (no GPU)."""
    from chat import CompressedChatModel
    print("\n" + "=" * 60)
    print("MODE: COMPRESSED CPU (C kernel)")
    print("=" * 60)

    _gc()
    rss_base = _rss_mb()
    _reset_vram_peak()
    vram_base = _vram_mb()

    print(f"  Baseline  — RSS: {rss_base:.0f} MB  VRAM: {vram_base:.0f} MB")
    print("  Loading model...", flush=True)
    t_load = time.time()

    model = CompressedChatModel(model_path, device='cpu', compression_mode='lossless')
    model.load()
    ids, tok = _tokenize(model_path, prompt)

    t_load = time.time() - t_load
    _gc()
    rss_after_load = _rss_mb()
    vram_after_load = _vram_mb()
    print(f"  After load — RSS: {rss_after_load:.0f} MB (+{rss_after_load - rss_base:.0f})  "
          f"VRAM: {vram_after_load:.0f} MB (+{vram_after_load - vram_base:.0f})  "
          f"load: {t_load:.1f}s")

    _reset_vram_peak()
    rss_pre_gen = _rss_mb()

    print(f"  Generating {max_tokens} tokens...", flush=True)
    model.model.eval()
    with torch.no_grad():
        t0 = time.time()
        out = model.model.generate(
            input_ids=ids, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0, pad_token_id=tok.pad_token_id,
        )
        gen_time = time.time() - t0

    new_tokens = out.shape[1] - ids.shape[1]
    tps = new_tokens / gen_time
    rss_post_gen = _rss_mb()
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    print(f"  Generation — RSS: {rss_post_gen:.0f} MB (Δ{rss_post_gen - rss_pre_gen:+.0f})  "
          f"VRAM peak: 0 MB  {tps:.2f} tok/s")
    print(f"  Text: \"{text}\"")

    del model; _gc()
    return dict(mode="Compressed CPU",
                rss_base=rss_base, rss_load=rss_after_load - rss_base,
                rss_gen_delta=rss_post_gen - rss_pre_gen,
                vram_load=0, vram_peak_gen=0,
                tps=tps, load_s=t_load, text=text)


def print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    hdr = f"{'':28s}  {'Uncomp GPU':>12s}  {'Compr GPU':>12s}  {'Compr CPU':>12s}"
    print(hdr)
    print("-" * 70)

    def row(label, key, fmt=".0f", unit=""):
        vals = []
        for r in results:
            if r.get('oom') and key not in ('mode',):
                vals.append("OOM")
            else:
                v = r.get(key, 0)
                vals.append(f"{v:{fmt}}{unit}" if v is not None else "n/a")
        print(f"  {label:26s}  {vals[0]:>12s}  {vals[1]:>12s}  {vals[2]:>12s}")

    row("CPU RAM added (load)",   "rss_load",      fmt="+.0f", unit=" MB")
    row("CPU RAM Δ (generation)", "rss_gen_delta", fmt="+.0f", unit=" MB")
    row("VRAM added (load)",      "vram_load",     fmt="+.0f", unit=" MB")
    row("VRAM peak (generation)", "vram_peak_gen", fmt=".0f",  unit=" MB")
    row("Load time",              "load_s",        fmt=".1f",  unit="s")
    row("Tokens/sec",             "tps",           fmt=".2f",  unit=" tok/s")

    print()
    print("  Generated text:")
    for r in results:
        print(f"    [{r['mode']:18s}]  \"{r['text']}\"")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default=str(MODEL_PATH))
    parser.add_argument("--prompt", default="Write a haiku about compression")
    parser.add_argument("--tokens", type=int, default=25)
    parser.add_argument("--mode", choices=["all", "gpu", "cpu", "compressed"],
                        default="all",
                        help="all=all three, gpu=uncomp+compr-gpu, cpu=compr-gpu+compr-cpu, "
                             "compressed=compr-gpu+compr-cpu")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    results = []

    run_unc = args.mode in ("all", "gpu")
    run_cgpu = args.mode in ("all", "gpu", "cpu", "compressed")
    run_ccpu = args.mode in ("all", "cpu", "compressed")

    if run_unc:
        results.append(run_uncompressed(model_path, args.prompt, args.tokens))
    if run_cgpu:
        results.append(run_compressed_gpu(model_path, args.prompt, args.tokens))
    if run_ccpu:
        results.append(run_compressed_cpu(model_path, args.prompt, args.tokens))

    if len(results) == 3:
        print_summary(results)
    elif len(results) == 2:
        # Pad to 3 for table alignment
        print_summary(results + [{"mode": "—", "rss_load": 0, "rss_gen_delta": 0,
                                   "vram_load": 0, "vram_peak_gen": 0,
                                   "tps": 0, "load_s": 0, "text": "—"}])


if __name__ == "__main__":
    main()
