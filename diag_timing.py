#!/usr/bin/env python3
"""Quick forward-pass timing: non-Huffman vs Huffman lossless."""
import sys, time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chat import CompressedChatModel

MODEL_PATH = '/home/bigattichouse/workspace/model/Qwen3.5-9B'
PROMPT = "What is the capital of France?"

def run_forward(model_obj, ids):
    """Time a single forward pass, return (elapsed_s, top5_tokens, top5_logits)."""
    tok = model_obj.tokenizer
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = model_obj.model(ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    logits = out.logits[0, -1].float()
    top5 = logits.topk(5)
    top5_tokens = [tok.decode([i]) for i in top5.indices.tolist()]
    return elapsed, top5_tokens, top5.values.tolist()

def vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

results = []

for label, entropy in [("Non-Huffman lossless", False), ("Huffman lossless", True)]:
    print(f"\n{'='*60}")
    print(f"Loading: {label}")
    chat = CompressedChatModel(
        MODEL_PATH, device='cuda',
        compression_mode='lossless',
        entropy_code=entropy,
    )
    t_load = time.time()
    chat.load()
    load_time = time.time() - t_load

    tok = chat.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors='pt').input_ids
    if torch.cuda.is_available():
        ids = ids.cuda()

    # Warm-up pass
    run_forward(chat, ids)

    # Timed pass
    elapsed, top5_tokens, top5_logits = run_forward(chat, ids)
    vram = vram_gb()

    print(f"  Load time   : {load_time:.1f}s")
    print(f"  Forward pass: {elapsed:.3f}s")
    print(f"  VRAM (peak) : {vram:.2f} GB")
    print(f"  Top-5 tokens: {top5_tokens}")
    print(f"  Next token  : {repr(top5_tokens[0])}")
    results.append((label, load_time, elapsed, vram, top5_tokens[0]))

    del chat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"{'Mode':<25} {'Load':>8} {'Forward':>10} {'VRAM':>9} {'Top-1':>15}")
print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*9} {'-'*15}")
for label, lt, fw, vr, tok in results:
    print(f"{label:<25} {lt:>7.1f}s {fw:>9.3f}s {vr:>7.2f}GB {repr(tok):>15}")
