#!/usr/bin/env python3
"""
Test Huffman inference quality by forcing CPU-RAM path.
Compare CPU-RAM and GPU Phase 2 outputs for the same model.
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, 'src')

# Force CPU-RAM path by patching compressed_modules before import
import compressed_modules as cm
# Temporarily disable GPU Phase 2 by patching HIP_AVAILABLE
_orig_hip = cm.HIP_AVAILABLE
cm.HIP_AVAILABLE = False  # Force CPU-RAM for all Huffman layers

from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from model_loader import CompressedModelLoader
from adaptive_compressor import AdaptiveCompressor

MODEL_PATH = Path('/home/bigattichouse/workspace/model/Qwen3.5-9B')
PROMPT = "What is the capital of France? Answer in one sentence."
MAX_TOKENS = 15

def load_model(use_hip=False):
    cm.HIP_AVAILABLE = use_hip
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if hasattr(config, 'text_config') and config.text_config is not None:
        for key, val in vars(config.text_config).items():
            if not hasattr(config, key):
                setattr(config, key, val)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compressor = AdaptiveCompressor(
        MODEL_PATH, compression_mode='lossless',
        store_in_model=True, entropy_code=True,
    )
    loader = CompressedModelLoader(
        MODEL_PATH, config, compressor,
        device='cuda', use_mmap=False,
    )

    t0 = time.time()
    model = loader.load()
    print(f"  Loaded in {time.time()-t0:.1f}s  GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model, tokenizer

def generate(model, tokenizer, prompt, max_tokens):
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors='pt').input_ids.cuda()

    with torch.no_grad():
        gen = model.generate(
            ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = gen[0, ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)

print("=" * 60)
print("Testing CPU-RAM path (use_gpu=False)")
print("=" * 60)
model_cpu, tok = load_model(use_hip=False)
resp_cpu = generate(model_cpu, tok, PROMPT, MAX_TOKENS)
print(f"CPU-RAM response: {repr(resp_cpu)}")

del model_cpu
torch.cuda.empty_cache()

print()
print("=" * 60)
print("Testing GPU Phase 2 path (use_gpu=True)")
print("=" * 60)
model_gpu, _ = load_model(use_hip=_orig_hip)
resp_gpu = generate(model_gpu, _, PROMPT, MAX_TOKENS)
print(f"GPU P2 response:  {repr(resp_gpu)}")

print()
print(f"CPU-RAM: {repr(resp_cpu)}")
print(f"GPU P2:  {repr(resp_gpu)}")
print("Match:", resp_cpu == resp_gpu)
