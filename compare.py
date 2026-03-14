#!/usr/bin/env python3
"""
compare.py — Side-by-side comparison of uncompressed vs. compressed inference.

Runs both modes on the same prompt, measures peak RAM and VRAM for each,
then prints a clean summary table.  Memory is thoroughly purged between runs
so the two measurements are independent.

For large models (7B+) the uncompressed run uses device_map="auto" which will
offload layers to CPU when VRAM is full.  This can be slow but avoids an OOM
crash.  Use --skip-uncompressed if you only want to measure the compressed run.

The compressed run requires a pre-built codebook cache.  Build it first:
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B

Usage:
    # Full comparison (uncompressed first, then compressed):
    ./venv/bin/python proofofconcept/compare.py ~/workspace/model/Qwen3.5-9B

    # With custom prompt and token count:
    ./venv/bin/python proofofconcept/compare.py ~/workspace/model/Qwen3.5-9B \\
        --prompt "Write a haiku about data compression" --tokens 80

    # Compressed-only (skip potentially slow/dangerous uncompressed run):
    ./venv/bin/python proofofconcept/compare.py ~/workspace/model/Qwen3.5-9B \\
        --skip-uncompressed
"""

import sys
import gc
import os
import time
import threading
import argparse
from pathlib import Path
from typing import Optional

import psutil
import torch
import numpy as np


# ─── output tee ──────────────────────────────────────────────────────────────

class _Tee:
    """Write to stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        self._stdout = sys.stdout
        self._file = open(log_path, 'w', encoding='utf-8', buffering=1)

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

    # Delegate attribute access so libraries that inspect sys.stdout still work.
    def __getattr__(self, name):
        return getattr(self._stdout, name)

# ─── memory helpers ──────────────────────────────────────────────────────────

def _ram_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9

def _vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

def _vram_reserved_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return 0.0

class PeakMemoryTracker:
    """Background thread that samples RAM+VRAM every 100 ms."""
    def __init__(self):
        self.peak_ram = _ram_gb()
        self.peak_vram = _vram_gb()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            self.peak_ram = max(self.peak_ram, _ram_gb())
            self.peak_vram = max(self.peak_vram, _vram_gb())
            time.sleep(0.1)

def purge_memory(label: str = ""):
    """Best-effort memory purge between runs."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset CUDA peak stats so the next run gets a clean high-water mark.
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    if label:
        ram = _ram_gb()
        vram = _vram_gb()
        print(f"  [purge {label}] RAM {ram:.2f} GB | VRAM {vram:.2f} GB")

# ─── uncompressed run ─────────────────────────────────────────────────────────

def run_uncompressed(model_path: Path, prompt: str, max_tokens: int,
                     temperature: float, device: str) -> dict:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    print(f"\n{'='*60}")
    print("UNCOMPRESSED (baseline)")
    print(f"{'='*60}")

    ram0 = _ram_gb()
    vram0 = _vram_gb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracker = PeakMemoryTracker().start()

    # Load
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, 'text_config'):
        for k, v in vars(config.text_config).items():
            if not hasattr(config, k):
                setattr(config, k, v)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",          # spans CPU+VRAM as needed
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t_load
    ram_after_load = _ram_gb()
    vram_after_load = _vram_gb()
    print(f"  Load : {load_time:.1f}s | RAM {ram_after_load:.2f} GB | VRAM {vram_after_load:.2f} GB")

    # Tokenize
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            if hasattr(input_ids, 'input_ids'):
                input_ids = input_ids.input_ids
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(next(model.parameters()).device)

    # Generate
    t_gen = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t_gen
    new_tokens = out.shape[1] - input_ids.shape[1]
    tps = new_tokens / gen_time if gen_time > 0 else 0.0

    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    tracker.stop()

    # Use CUDA's built-in peak tracker (catches spikes between sampler intervals).
    cuda_peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    peak_vram = max(tracker.peak_vram, cuda_peak_vram)

    print(f"  Gen  : {gen_time:.1f}s | {tps:.1f} tok/s | {new_tokens} tokens")
    print(f"  Peak : RAM {tracker.peak_ram:.2f} GB | VRAM {peak_vram:.2f} GB")
    print(f"\n  Response: {text}")

    result = {
        'mode': 'uncompressed',
        'load_time': load_time,
        'gen_time': gen_time,
        'tps': tps,
        'new_tokens': new_tokens,
        'ram_load': ram_after_load,
        'vram_load': vram_after_load,
        'peak_ram': tracker.peak_ram,
        'peak_vram': peak_vram,
        'text': text,
    }

    # Thorough cleanup
    del model, tokenizer, input_ids, out
    return result

# ─── compressed run ───────────────────────────────────────────────────────────

def run_compressed(model_path: Path, prompt: str, max_tokens: int,
                   temperature: float, device: str,
                   compression_mode: str = 'lossless') -> Optional[dict]:
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    sys.path.insert(0, str(Path(__file__).parent))
    from chat import CompressedChatModel

    print(f"\n{'='*60}")
    print("COMPRESSED (codebook inference)")
    print(f"{'='*60}")

    # Check cache
    cache_tensors = model_path / 'codebook' / 'tensors'
    if not cache_tensors.exists() or len(list(cache_tensors.glob('*.npz'))) == 0:
        print(f"\n  No compression cache found.")
        print(f"  Run first: ./venv/bin/python proofofconcept/compress.py {model_path}")
        return None

    ram0 = _ram_gb()
    vram0 = _vram_gb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracker = PeakMemoryTracker().start()

    t_load = time.time()
    print(f"  Compression mode: {compression_mode}")
    chat_model = CompressedChatModel(
        model_path, device=device, compression_mode=compression_mode
    )
    if not chat_model.load():
        tracker.stop()
        return None
    load_time = time.time() - t_load
    ram_after_load = _ram_gb()
    vram_after_load = _vram_gb()
    print(f"  Load : {load_time:.1f}s | RAM {ram_after_load:.2f} GB | VRAM {vram_after_load:.2f} GB")

    t_gen = time.time()
    text = chat_model.generate(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    gen_time = time.time() - t_gen
    # generate() reports its own tok/s — estimate from time
    # (CompressedChatModel.generate prints it internally)
    tracker.stop()

    # Re-measure tps from the timing (generate() returns text, not token count)
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    new_tokens = len(_tok.encode(text))
    tps = new_tokens / gen_time if gen_time > 0 else 0.0
    del _tok

    cuda_peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    peak_vram = max(tracker.peak_vram, cuda_peak_vram)

    print(f"  Gen  : {gen_time:.1f}s | {tps:.1f} tok/s | {new_tokens} tokens")
    print(f"  Peak : RAM {tracker.peak_ram:.2f} GB | VRAM {peak_vram:.2f} GB")
    print(f"\n  Response: {text}")

    result = {
        'mode': 'compressed',
        'load_time': load_time,
        'gen_time': gen_time,
        'tps': tps,
        'new_tokens': new_tokens,
        'ram_load': ram_after_load,
        'vram_load': vram_after_load,
        'peak_ram': tracker.peak_ram,
        'peak_vram': peak_vram,
        'text': text,
    }

    del chat_model
    return result

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare uncompressed vs compressed inference'
    )
    parser.add_argument('model_path', help='Model directory')
    parser.add_argument('--prompt', default='Write a haiku about data compression',
                        help='Prompt for both runs')
    parser.add_argument('--tokens', type=int, default=80,
                        help='Max new tokens (default: 80)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0.0 = greedy, reproducible)')
    parser.add_argument('--device', default='cuda',
                        help='Device for compressed run (default: cuda)')
    parser.add_argument('--skip-uncompressed', action='store_true',
                        help='Skip uncompressed run (useful if model is too large)')
    parser.add_argument('--mode', default='lossless',
                        choices=['lossless', 'balanced', 'aggressive'],
                        help='Compression mode for compressed run (default: lossless)')
    parser.add_argument('--log', default=None,
                        help='Path to save full output log (default: auto-named in cwd)')
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        print(f"Error: {model_path} does not exist")
        sys.exit(1)

    # Set up output tee (log to file + stdout simultaneously)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.log:
        log_path = Path(args.log)
    else:
        log_dir = Path(__file__).parent / "comparison"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{model_path.name}_{args.mode}_{timestamp}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging to: {log_path}")

    print(f"\nModel : {model_path.name}")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Tokens: {args.tokens}")

    # Baseline RAM
    ram_baseline = _ram_gb()
    vram_baseline = _vram_gb()
    print(f"\nBaseline: RAM {ram_baseline:.2f} GB | VRAM {vram_baseline:.2f} GB")

    results = {}

    # ── uncompressed ──
    if not args.skip_uncompressed:
        results['uncompressed'] = run_uncompressed(
            model_path, args.prompt, args.tokens, args.temperature, args.device
        )
        purge_memory("between runs")
        time.sleep(1.0)  # give OS a moment to reclaim pages
    else:
        print("\n[Skipping uncompressed run]")

    # ── compressed ──
    results['compressed'] = run_compressed(
        model_path, args.prompt, args.tokens, args.temperature, args.device, args.mode
    )
    purge_memory("after compressed")

    # ── summary ──
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Prompt: \"{args.prompt}\"")
    print()

    uc = results.get('uncompressed')
    co = results.get('compressed')

    w = 22
    header = f"{'Metric':<24} {'Uncompressed':>{w}} {'Compressed':>{w}}"
    print(header)
    print('-' * len(header))

    def row(label, uc_val, co_val, fmt="{:.2f}", suffix=""):
        u = (fmt.format(uc_val) + suffix) if uc_val is not None else "skipped"
        c = (fmt.format(co_val) + suffix) if co_val is not None else "n/a"
        print(f"{label:<24} {u:>{w}} {c:>{w}}")

    row("Load time (s)",
        uc['load_time'] if uc else None,
        co['load_time'] if co else None, suffix="s")
    row("Peak RAM (GB)",
        uc['peak_ram'] if uc else None,
        co['peak_ram'] if co else None, suffix=" GB")
    row("Peak VRAM (GB)",
        uc['peak_vram'] if uc else None,
        co['peak_vram'] if co else None, suffix=" GB")
    row("Speed (tok/s)",
        uc['tps'] if uc else None,
        co['tps'] if co else None, fmt="{:.1f}", suffix=" tok/s")

    print()

    if uc and co:
        ram_delta = uc['peak_ram'] - co['peak_ram']
        vram_delta = uc['peak_vram'] - co['peak_vram']
        print(f"  RAM  savings : {ram_delta:+.2f} GB  (compressed uses {ram_delta/uc['peak_ram']*100:.1f}% less)")
        print(f"  VRAM savings : {vram_delta:+.2f} GB")
        speed_ratio = co['tps'] / uc['tps'] if uc['tps'] > 0 else float('nan')
        print(f"  Speed ratio  : {speed_ratio:.2f}x  (compressed / uncompressed)")

    print()
    print("─── Uncompressed response ───────────────────────────────")
    print(uc['text'] if uc else "(skipped)")
    print()
    print("─── Compressed response ─────────────────────────────────")
    print(co['text'] if co else "(n/a — run compress.py first)")
    print()

    tee.close()
    # Print log path to the real stdout after restoring it.
    print(f"\nFull log saved to: {log_path}", file=sys.__stdout__)


if __name__ == '__main__':
    main()
