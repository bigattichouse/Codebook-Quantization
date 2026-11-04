"""
gguf_entropy.py — CPU-only lossless-headroom survey of an already-quantized GGUF.

Huffman-codes the integer quant codes (not the dequantized floats) per tensor
and reports the size saving, plus shared-dictionary and code-pair / 4-code-word
context tests.  See src/gguf_codes.py.

Usage:
    ./venv/bin/python analysis/gguf_entropy.py ../gguf/Qwen3.8-Flash-Next
    ./venv/bin/python analysis/gguf_entropy.py model.gguf --blocks 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from gguf_codes import print_headroom, scan_gguf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="GGUF file or directory of shards")
    ap.add_argument("--blocks", type=int, default=8192,
                    help="max blocks sampled per tensor (default 8192)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = scan_gguf(args.path, max_blocks=args.blocks, seed=args.seed)
    print()
    print(f"  LOSSLESS HEADROOM (per-tensor Huffman on quant codes): {args.path.name}")
    print_headroom(res, context=True, roles=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
