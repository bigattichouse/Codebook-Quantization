#!/usr/bin/env python3
"""
eval.py — Perplexity evaluation for compressed vs uncompressed models.

Perplexity measures how well the model predicts a fixed test passage.
Lower = better.  A pure INT8 linear-quantized 9B model typically adds
~0.05–0.15 perplexity above the uncompressed baseline.  We use this
as the quality bar for balanced compression.

Usage:
    python proofofconcept/eval.py ~/workspace/model/Qwen3.5-9B --mode lossless
    python proofofconcept/eval.py ~/workspace/model/Qwen3.5-9B --mode balanced
    python proofofconcept/eval.py ~/workspace/model/Qwen3.5-9B --mode uncompressed

Compare the PPL values across modes:
    uncompressed  → baseline
    lossless      → should equal baseline (codebook is exact for lossless)
    balanced      → small increase acceptable (target: < +0.5 vs uncompressed)

Options:
    --stride    Sliding window stride in tokens (default: 256)
    --context   Tokens of context per window (default: 512)
    --tokens    Max tokens of test text to evaluate (default: 2048)
"""

import sys
import time
import math
import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from memory_utils import resolve_device

# ---------------------------------------------------------------------------
# Fixed test passage — used for all runs so numbers are directly comparable.
# Mix of factual prose, technical content, and narrative.
# ---------------------------------------------------------------------------

TEST_TEXT = """
The transformer architecture, introduced in 2017, relies on self-attention
mechanisms to model relationships between tokens in a sequence. Unlike
recurrent networks, transformers process all tokens in parallel, enabling
efficient training on modern hardware. The key insight is that each token
can attend to every other token with a weight proportional to their
relevance, computed via dot-product similarity between learned query and
key vectors.

Data compression exploits statistical redundancy in information. The
fundamental limit is given by Shannon's entropy: no lossless scheme can
compress a source below its entropy rate. Practical compressors like gzip
use a combination of LZ77 sliding-window matching and Huffman coding.
Neural network weights, being floating-point tensors with characteristic
distributions clustered near zero, compress well with vector quantization
approaches that replace similar values with a shared codebook entry.

Large language models have demonstrated remarkable capabilities across
tasks including translation, summarization, code generation, and
mathematical reasoning. The scaling laws observed by Kaplan et al. suggest
that model performance improves predictably with compute, data, and
parameter count. However, the quadratic complexity of standard attention
with respect to sequence length remains a practical constraint for very
long contexts, motivating research into linear-complexity alternatives.

Efficient inference is critical for deploying large models in production.
Memory bandwidth is often the limiting factor: at decode time, each
generated token requires reading all model weights from GPU memory. A
70-billion-parameter model in bfloat16 occupies 140 GB, far exceeding
the memory of a single consumer GPU. Quantization to 4-bit precision
reduces this to 35 GB while maintaining most of the model's quality,
enabling deployment on commodity hardware with 48 GB of VRAM.

The KV cache stores key and value tensors for all previously processed
tokens, avoiding redundant computation during autoregressive generation.
Its size grows linearly with context length: for a model with 32 layers,
8 KV heads, and a head dimension of 128, each token requires 32 × 2 × 8
× 128 × 2 bytes ≈ 131 KB of cache. At 100,000 tokens of context, this
amounts to approximately 13 GB — a significant fraction of available GPU
memory that can be halved by storing activations in 8-bit precision
instead of bfloat16, at minimal quality cost.

Retrieval-augmented generation combines a parametric language model with
a non-parametric retrieval system. Given a query, relevant documents are
fetched from a large corpus and prepended to the prompt, allowing the
model to answer questions about information not present in its training
data. The quality of the retrieval step is critical: irrelevant or
misleading documents can degrade model performance significantly.

Computer architecture evolution has been driven by the end of Dennard
scaling and the slowdown of Moore's law. Modern chips compensate through
specialization: GPUs provide thousands of parallel arithmetic units for
dense matrix operations, while custom ASICs like TPUs offer even higher
throughput for the specific workloads encountered in deep learning. Memory
bandwidth has become the primary bottleneck for large model inference,
favoring architectures with high-bandwidth memory such as HBM2 and HBM3.
""".strip()


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, text: str,
                       context: int = 512,
                       stride: int = 256,
                       device: str = 'cuda') -> dict:
    """
    Sliding-window perplexity.  Each window of `context` tokens is evaluated
    with `stride` tokens of fresh output per window (the first context-stride
    tokens are conditioning only).  This avoids boundary effects for long texts.
    """
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0
    prev_end = 0
    t0 = time.time()

    for begin in range(0, seq_len, stride):
        end = min(begin + context, seq_len)
        target_len = end - max(begin, prev_end)  # tokens scored this window

        chunk = input_ids[:, begin:end]
        target = chunk.clone()
        # Mask the conditioning tokens (not scored)
        target[:, :-target_len] = -100

        with torch.no_grad():
            out = model(chunk, labels=target)
            nll = out.loss.item() * target_len

        nlls.append(nll)
        n_tokens += target_len
        prev_end = end

        if end >= seq_len:
            break

    elapsed = time.time() - t0
    total_nll = sum(nlls)
    ppl = math.exp(total_nll / n_tokens)
    return {
        'ppl':      round(ppl, 4),
        'nll':      round(total_nll / n_tokens, 4),
        'n_tokens': n_tokens,
        'elapsed_s': round(elapsed, 2),
        'tok_per_s': round(n_tokens / elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Model loading (reuses chat.py infrastructure)
# ---------------------------------------------------------------------------

def load_model(model_path: Path, mode: str, device: str):
    from uncompressed_loader import UncompressedKernelLoader

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, 'text_config') and config.text_config is not None:
        for k, v in vars(config.text_config).items():
            if not hasattr(config, k):
                setattr(config, k, v)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16

    if mode == 'uncompressed':
        loader = UncompressedKernelLoader(device=device)
        model = loader.load(model_path, config=config, dtype=dtype)
        return model, tokenizer

    # Compressed modes
    from adaptive_compressor import AdaptiveCompressor
    from model_loader import CompressedModelLoader
    from name_resolver import NameResolver

    mse_target = {'lossless': 0.0, 'balanced': 0.005}.get(mode, 0.005)
    compressor = AdaptiveCompressor(
        model_path, compression_mode=mode, store_in_model=True,
        mse_threshold=mse_target,
    )
    _, metadata = compressor.load_compressed(load_tensors=True)

    codebooks = {}
    for ttype, cb in metadata.get('global_codebooks', {}).items():
        codebooks[ttype] = cb.to(device=device, dtype=torch.float32)

    from transformers import AutoModelForCausalLM
    with torch.device('meta'):
        meta = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    resolver = NameResolver.from_model_and_compressor(meta, compressor)

    loader = CompressedModelLoader(
        model_path=model_path, device=device,
        compressor=compressor, codebooks=codebooks,
    )
    model = loader.create_and_load(config, dtype, resolver)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Perplexity evaluation')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--mode', default='uncompressed',
                        choices=['uncompressed', 'lossless', 'balanced'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--context', type=int, default=512,
                        help='Tokens of context per window (default: 512)')
    parser.add_argument('--stride', type=int, default=256,
                        help='Sliding window stride (default: 256)')
    parser.add_argument('--tokens', type=int, default=2048,
                        help='Max test tokens to evaluate (default: 2048)')
    args = parser.parse_args()

    # Match chat.py's device handling: PyTorch tensors fall back to CPU when
    # CUDA/ROCm isn't visible to PyTorch.  GPU acceleration still happens via
    # direct HIP ctypes kernels inside AdaptiveCodebookLinear — those work
    # regardless of the PyTorch device.
    args.device = resolve_device(args.device)

    model_path = args.model_path.expanduser()

    print(f"\nMode: {args.mode}  |  Device: {args.device}")
    print(f"Context: {args.context}  |  Stride: {args.stride}  |  Max tokens: {args.tokens}")
    print("Loading model...")
    t_load = time.time()
    model, tokenizer = load_model(model_path, args.mode, args.device)
    load_s = time.time() - t_load
    print(f"Loaded in {load_s:.1f}s")

    # Trim test text to requested token budget
    enc = tokenizer(TEST_TEXT, return_tensors='pt')
    n_avail = enc.input_ids.shape[1]
    if n_avail > args.tokens:
        trimmed = tokenizer.decode(enc.input_ids[0, :args.tokens], skip_special_tokens=True)
    else:
        trimmed = TEST_TEXT
    print(f"Test text: {min(n_avail, args.tokens)} tokens")

    print("Computing perplexity...")
    result = compute_perplexity(
        model, tokenizer, trimmed,
        context=args.context,
        stride=args.stride,
        device=args.device,
    )

    print(f"\n{'='*55}")
    print(f"Mode        : {args.mode}")
    print(f"Perplexity  : {result['ppl']}")
    print(f"NLL         : {result['nll']}")
    print(f"Tokens eval : {result['n_tokens']}")
    print(f"Eval speed  : {result['tok_per_s']} tok/s")
    print(f"Load time   : {load_s:.1f}s")
    print(f"{'='*55}")
    print(f"\nFor comparison — typical INT8 linear quant adds ~0.05–0.15 PPL")
    print(f"above uncompressed baseline on a 9B model.")

    # --- Sanity-check generation: show actual output on fixed prompts --------
    print(f"\n{'='*55}")
    print("GENERATION SANITY CHECK")
    print(f"{'='*55}")
    prompts = [
        "Write a haiku about data compression.",
        "What is the capital of France?",
        "Complete this sentence: The transformer architecture works by",
    ]
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(args.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"\nPrompt : {prompt}")
        print(f"Output : {response.strip()}")
    print(f"\n{'='*55}")


if __name__ == '__main__':
    main()
