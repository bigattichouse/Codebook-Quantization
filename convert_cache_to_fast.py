#!/usr/bin/env python3
"""
convert_cache_to_fast.py — Re-export the compression cache as uncompressed .npz.

The default cache uses np.savez_compressed() (gzip).  Loading 775 files
takes ~18 s because gzip decompression is CPU-bound even with 8 threads.

This script converts each .npz to an uncompressed .npz (plain zip, no deflate)
which np.load() reads 4–6× faster at the cost of ~15–30% more disk space.

Usage:
    python proofofconcept/convert_cache_to_fast.py ~/workspace/model/Qwen3.5-9B

The originals are kept with a .npz.gz backup extension.  Re-run with --restore
to revert to the compressed originals.

Options:
    --restore     Restore .npz.gz backups and delete the fast .npz files
    --no-backup   Skip creating .npz.gz backups (saves disk space, irreversible)
    --dry-run     Print what would be done without changing any files
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def convert(model_dir: Path, backup: bool = True, dry_run: bool = False):
    tensors_dir = model_dir / "codebook" / "tensors"
    if not tensors_dir.exists():
        print(f"Error: no cache found at {tensors_dir}")
        sys.exit(1)

    npz_files = sorted(tensors_dir.glob("*.npz"))
    if not npz_files:
        print("No .npz files found — nothing to do.")
        return

    # Skip files that are already uncompressed (check first file)
    def _is_compressed(p: Path) -> bool:
        import zipfile
        try:
            with zipfile.ZipFile(p) as z:
                return any(i.compress_type != 0 for i in z.infolist())
        except Exception:
            return True  # assume compressed on error

    already_fast = sum(1 for f in npz_files if not _is_compressed(f))
    if already_fast == len(npz_files):
        print(f"All {len(npz_files)} files are already uncompressed — nothing to do.")
        return
    if already_fast:
        print(f"Note: {already_fast}/{len(npz_files)} already uncompressed, converting the rest.")

    total_before = sum(f.stat().st_size for f in npz_files)
    print(f"Converting {len(npz_files)} files in {tensors_dir}")
    print(f"Disk before: {total_before / 1e9:.2f} GB")
    if dry_run:
        print("[dry-run] No files will be modified.")
        return

    t0 = time.time()
    for i, npz_path in enumerate(npz_files, 1):
        if not _is_compressed(npz_path):
            continue  # already fast

        # Load
        data = dict(np.load(npz_path, allow_pickle=False))

        # Backup
        if backup:
            bak = npz_path.with_suffix(".npz.gz")
            if not bak.exists():
                shutil.copy2(npz_path, bak)

        # Write uncompressed
        tmp = npz_path.with_suffix(".npz.tmp")
        np.savez(tmp, **data)       # savez = uncompressed zip
        tmp.replace(npz_path)

        if i % 50 == 0 or i == len(npz_files):
            elapsed = time.time() - t0
            print(f"  [{i:4d}/{len(npz_files)}] {elapsed:.1f}s elapsed", flush=True)

    total_after = sum(f.stat().st_size for f in npz_files)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Disk after:  {total_after / 1e9:.2f} GB  "
          f"(delta: +{(total_after - total_before)/1e9:.2f} GB)")
    if backup:
        bak_size = sum(f.stat().st_size for f in tensors_dir.glob("*.npz.gz"))
        print(f"Backups:     {bak_size / 1e9:.2f} GB  (delete with --no-backup next run)")
    print("Expected load time: ~3–4 s  (was ~18 s)")


def restore(model_dir: Path, dry_run: bool = False):
    tensors_dir = model_dir / "codebook" / "tensors"
    bak_files = sorted(tensors_dir.glob("*.npz.gz"))
    if not bak_files:
        print("No .npz.gz backups found — nothing to restore.")
        return
    print(f"Restoring {len(bak_files)} backup files...")
    if dry_run:
        print("[dry-run] No files will be modified.")
        return
    for bak in bak_files:
        orig = bak.with_suffix("")   # strips .gz → .npz
        shutil.copy2(bak, orig)
        bak.unlink()
    print(f"Restored {len(bak_files)} files.  Load time returns to ~18 s.")


def main():
    parser = argparse.ArgumentParser(description="Convert compression cache to fast (uncompressed) format")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--restore", action="store_true", help="Restore .npz.gz backups")
    parser.add_argument("--no-backup", action="store_true", help="Skip .npz.gz backups")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without changing files")
    args = parser.parse_args()

    if args.restore:
        restore(args.model_dir.expanduser(), dry_run=args.dry_run)
    else:
        convert(args.model_dir.expanduser(), backup=not args.no_backup, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
