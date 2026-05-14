"""Verify that a Wang Corel dataset is laid out correctly on disk.

Usage:
    PYTHONPATH=. python scripts/verify_dataset.py --data data/corel1k --variant 1k

This is a sanity check, not a downloader. Wang Corel doesn't have a single
canonical source — see DATASETS.md for where to get it.
"""
import argparse
from pathlib import Path
import numpy as np

from src.utils.dataset import CorelDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--variant", choices=["1k", "10k"], default="1k")
    args = p.parse_args()

    expected_total = 1000 if args.variant == "1k" else 10000

    try:
        ds = CorelDataset(args.data, variant=args.variant)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print(f"✓ Found {len(ds)} images at {args.data}")

    if len(ds) != expected_total:
        print(f"⚠ Expected {expected_total} images for Corel-{args.variant}, "
              f"got {len(ds)}. The loader will still work but evaluation "
              f"numbers may not match published benchmarks.")

    counts = np.bincount(ds.labels)
    print(f"✓ Detected {ds.num_classes} classes")
    print(f"✓ Class distribution: min={counts.min()}, max={counts.max()}, "
          f"median={int(np.median(counts))}")

    # Try to load the first image — catches corrupted files early
    try:
        img = ds[0].load()
        print(f"✓ First image loads OK, shape = {img.shape}")
    except Exception as e:
        print(f"❌ Failed to load first image: {e}")
        return

    print(f"\nReady to build an index:\n"
          f"  python scripts/build_index.py "
          f"--data {args.data} --variant {args.variant} "
          f"--backend classical --out indexes/classical_{args.variant}")


if __name__ == "__main__":
    main()
