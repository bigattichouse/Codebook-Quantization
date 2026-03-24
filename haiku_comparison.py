#!/usr/bin/env python3
"""
Three-way comparison: uncompressed / codebook / codebook+huffman
Prompt: "Write a haiku about data compression"
Reports disk size, VRAM, CPU RAM, load time, inference time, and the haiku.
"""
import sys, time, gc
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chat import CompressedChatModel

MODEL_PATH = '/home/bigattichouse/workspace/model/Qwen3.5-9B'
PROMPT = "Write a haiku about data compression"
MAX_TOKENS = 80

DISK = {
    'uncompressed': 19,
    'codebook':     15,
    'huffman':      12,
}

def get_mem():
    import psutil
    cpu_gb = psutil.Process().memory_info().rss / 1e9
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return cpu_gb, vram_gb

def run_mode(label, mode, entropy):
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")

    chat = CompressedChatModel(
        MODEL_PATH, device='cuda',
        compression_mode=mode,
        entropy_code=entropy,
    )

    t_load = time.time()
    chat.load()
    load_time = time.time() - t_load

    cpu_after_load, vram_after_load = get_mem()

    tok = chat.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors='pt').input_ids.cuda()

    # Generate
    t_gen = time.time()
    with torch.no_grad():
        out_ids = chat.model.generate(
            ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_time = time.time() - t_gen

    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    cpu_peak, _ = get_mem()

    new_ids = out_ids[0, ids.shape[1]:]
    n_tokens = len(new_ids)
    response = tok.decode(new_ids, skip_special_tokens=True).strip()

    # Scrub <think>...</think> block if present
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    tok_per_sec = n_tokens / gen_time if gen_time > 0 else 0

    result = dict(
        label=label,
        disk_gb=DISK[entropy and 'huffman' or (mode == 'uncompressed' and 'uncompressed' or 'codebook')],
        load_time=load_time,
        vram_load=vram_after_load,
        vram_peak=vram_peak,
        cpu_ram=cpu_peak,
        gen_time=gen_time,
        n_tokens=n_tokens,
        tok_per_sec=tok_per_sec,
        response=response,
    )

    print(f"\n  Haiku:\n    {response.replace(chr(10), chr(10)+'    ')}")
    print(f"\n  Load: {load_time:.1f}s  |  Gen: {gen_time:.1f}s ({tok_per_sec:.1f} tok/s, {n_tokens} tokens)")
    print(f"  VRAM at load: {vram_after_load:.2f} GB  peak: {vram_peak:.2f} GB  |  CPU RAM: {cpu_peak:.2f} GB")

    del chat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


results = []
results.append(run_mode("Uncompressed GPU",         mode='uncompressed', entropy=False))
results.append(run_mode("Codebook GPU (lossless)",  mode='lossless',     entropy=False))
results.append(run_mode("Codebook + Huffman GPU",   mode='lossless',     entropy=True))

# ── Summary table ────────────────────────────────────────────────────────────
W = 26
print(f"\n\n{'='*90}")
print(f"{'SUMMARY':^90}")
print(f"{'='*90}")
hdr = f"{'Mode':<{W}} {'Disk':>6} {'VRAM':>7} {'CPU RAM':>8} {'Load':>7} {'Inf':>7} {'tok/s':>7}"
print(hdr)
print('-' * len(hdr))
for r in results:
    print(
        f"{r['label']:<{W}} "
        f"{r['disk_gb']:>5}GB "
        f"{r['vram_peak']:>6.2f}GB "
        f"{r['cpu_ram']:>7.2f}GB "
        f"{r['load_time']:>6.1f}s "
        f"{r['gen_time']:>6.1f}s "
        f"{r['tok_per_sec']:>6.1f}"
    )

print(f"\n{'─'*90}")
print("Haiku outputs:")
for r in results:
    print(f"\n  [{r['label']}]")
    for line in r['response'].splitlines():
        print(f"    {line}")
