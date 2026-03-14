#!/usr/bin/env python3
"""
Speed comparison: compressed vs. uncompressed inference.

Runs both uncompressed_chat.py and compressed chat.py on the same prompt,
then reports tokens/sec and memory for each.

Usage:
    ./venv/bin/python proofofconcept/tests/speed_compare.py ~/workspace/model/Qwen3.5-0.8B
    ./venv/bin/python proofofconcept/tests/speed_compare.py ~/workspace/model/Qwen3.5-0.8B \\
        --prompt "Explain quantum computing" --tokens 50
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_script(python: str, script: str, model_path: str, prompt: str, tokens: int):
    """Run a chat script and capture its output."""
    cmd = [
        python, script, model_path,
        "--prompt", prompt,
        "--max-tokens", str(tokens),
    ]
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    wall_time = time.time() - t0
    return result.stdout + result.stderr, wall_time


def parse_tps(output: str) -> float:
    """Extract tokens/sec from script output."""
    for line in output.splitlines():
        low = line.lower()
        if "tok/s" in low or "tokens/sec" in low or "tokens/second" in low:
            for token in line.replace(",", "").split():
                try:
                    val = float(token)
                    if 0.01 < val < 10000:
                        return val
                except ValueError:
                    continue
    return 0.0


def parse_memory(output: str) -> float:
    """Extract model memory usage (GB) from script output."""
    for line in output.splitlines():
        low = line.lower()
        if "memory usage" in low or "gpu ram" in low or "vram" in low:
            for token in line.replace(",", "").split():
                try:
                    val = float(token)
                    if 0.001 < val < 100:
                        return val
                except ValueError:
                    continue
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare compressed vs uncompressed inference")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    parser.add_argument("--prompt", default="Write a haiku about the ocean",
                        help="Prompt to use for both runs")
    parser.add_argument("--tokens", type=int, default=20,
                        help="Max tokens to generate")
    parser.add_argument("--skip-uncompressed", action="store_true",
                        help="Skip uncompressed run (e.g. if model too large for VRAM)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    python = sys.executable
    compressed_script = str(project_dir / "chat.py")
    uncompressed_script = str(project_dir / "uncompressed_chat.py")

    print(f"{'=' * 70}")
    print(f"SPEED COMPARISON")
    print(f"{'=' * 70}")
    print(f"Model:  {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Tokens: {args.tokens}")
    print(f"{'=' * 70}")
    print()

    results = {}

    # Uncompressed
    if not args.skip_uncompressed:
        print("--- Running UNCOMPRESSED inference ---")
        try:
            output_u, wall_u = run_script(
                python, uncompressed_script, args.model_path, args.prompt, args.tokens
            )
            tps_u = parse_tps(output_u)
            mem_u = parse_memory(output_u)
            results["uncompressed"] = {"tps": tps_u, "mem_gb": mem_u, "wall_s": wall_u}
            print(f"  tokens/sec: {tps_u:.1f}")
            print(f"  memory:     {mem_u:.2f} GB")
            print(f"  wall time:  {wall_u:.1f}s")
        except subprocess.TimeoutExpired:
            print("  TIMEOUT (>600s)")
            results["uncompressed"] = {"tps": 0, "mem_gb": 0, "wall_s": 600}
        except Exception as e:
            print(f"  ERROR: {e}")
            results["uncompressed"] = {"tps": 0, "mem_gb": 0, "wall_s": 0}
        print()

    # Compressed
    print("--- Running COMPRESSED inference ---")
    try:
        output_c, wall_c = run_script(
            python, compressed_script, args.model_path, args.prompt, args.tokens
        )
        tps_c = parse_tps(output_c)
        mem_c = parse_memory(output_c)
        results["compressed"] = {"tps": tps_c, "mem_gb": mem_c, "wall_s": wall_c}
        print(f"  tokens/sec: {tps_c:.1f}")
        print(f"  memory:     {mem_c:.2f} GB")
        print(f"  wall time:  {wall_c:.1f}s")
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (>600s)")
        results["compressed"] = {"tps": 0, "mem_gb": 0, "wall_s": 600}
    except Exception as e:
        print(f"  ERROR: {e}")
        results["compressed"] = {"tps": 0, "mem_gb": 0, "wall_s": 0}
    print()

    # Summary
    print(f"{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"{'':20s} {'Uncompressed':>15s} {'Compressed':>15s}")
    print(f"{'-' * 50}")
    u = results.get("uncompressed", {})
    c = results.get("compressed", {})
    print(f"{'tokens/sec':20s} {u.get('tps', 0):>15.1f} {c.get('tps', 0):>15.1f}")
    print(f"{'memory (GB)':20s} {u.get('mem_gb', 0):>15.2f} {c.get('mem_gb', 0):>15.2f}")
    print(f"{'wall time (s)':20s} {u.get('wall_s', 0):>15.1f} {c.get('wall_s', 0):>15.1f}")

    if u.get("tps", 0) > 0 and c.get("tps", 0) > 0:
        speedup = u["tps"] / c["tps"]
        print(f"\nSpeed ratio: uncompressed is {speedup:.1f}x faster")
    if u.get("mem_gb", 0) > 0 and c.get("mem_gb", 0) > 0:
        mem_ratio = u["mem_gb"] / c["mem_gb"]
        print(f"Memory ratio: compressed uses {mem_ratio:.1f}x less")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
