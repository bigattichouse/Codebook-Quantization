#!/usr/bin/env python3
"""Compare single forward pass between non-Huffman and Huffman lossless models."""
import sys, time, re
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chat import CompressedChatModel

MODEL_PATH = '/home/bigattichouse/workspace/model/Qwen3.5-9B'
PROMPT = "What is the capital of France?"

def load_model(entropy_code=False):
    chat = CompressedChatModel(
        MODEL_PATH, device='cuda',
        compression_mode='lossless',
        entropy_code=entropy_code,
    )
    t0 = time.time()
    chat.load()
    print(f"  Loaded in {time.time()-t0:.1f}s  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")
    return chat

tok_obj = None

def get_ids(chat):
    global tok_obj
    tok_obj = chat.tokenizer
    text = tok_obj.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    return tok_obj(text, return_tensors='pt').input_ids.cuda()

print(f"Input: {repr(PROMPT)}\n")

print("Loading NON-HUFFMAN lossless model...")
chat_base = load_model(entropy_code=False)
ids = get_ids(chat_base)
with torch.no_grad():
    out_base = chat_base.model(ids)
logits_base = out_base.logits[0, -1].float()
top5_base = logits_base.topk(5)
print(f"  Top-5 tokens: {[tok_obj.decode([i]) for i in top5_base.indices.tolist()]}")
print(f"  Next token  : {tok_obj.decode([logits_base.argmax().item()])!r}")

del chat_base
torch.cuda.empty_cache()
print()

print("Loading HUFFMAN lossless model...")
chat_huff = load_model(entropy_code=True)
ids = get_ids(chat_huff)
with torch.no_grad():
    t0 = time.time()
    out_huff = chat_huff.model(ids)
    torch.cuda.synchronize()
    print(f"  Forward pass: {time.time()-t0:.2f}s")
logits_huff = out_huff.logits[0, -1].float()
top5_huff = logits_huff.topk(5)
print(f"  Top-5 tokens: {[tok_obj.decode([i]) for i in top5_huff.indices.tolist()]}")
print(f"  Next token  : {tok_obj.decode([logits_huff.argmax().item()])!r}")

print()
match = logits_base.argmax().item() == logits_huff.argmax().item()
err = (logits_base - logits_huff).abs()
print(f"Top-1 match : {match}")
print(f"Logit max err: {err.max():.4f}  mean err: {err.mean():.4f}")
