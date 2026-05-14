"""Run a full retrieval evaluation with a chosen feature extractor.

Usage:
    PYTHONPATH=. python scripts/evaluate_classical.py \\
        --data data/corel1k --variant 1k --extractor combo

What it does:
  1. Loads the dataset.
  2. Extracts features for every image.
  3. Treats every image in turn as a query, retrieves the rest by cosine similarity.
  4. Prints P@K, R@K, and mAP.

On Corel-1K with histogram alone you should see mAP somewhere around 0.40–0.50.
With combo, around 0.50–0.60. Deep features will roughly double that — but
that's the next milestone.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.classical.extractors import get_extractor
from src.utils.dataset import CorelDataset
from src.utils.metrics import evaluate_retrieval, format_metrics_table


def build_feature_matrix(dataset: CorelDataset, extractor) -> np.ndarray:
    """Extract a (N, D) feature matrix for the whole dataset."""
    feats = np.zeros((len(dataset), extractor.dim), dtype=np.float32)
    for i, record in enumerate(tqdm(dataset, desc=f"extract:{extractor.name}")):
        # cv2.imread returns BGR — perfect for our extractors
        img_bgr = cv2.imread(str(record.path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {record.path}")
        feats[i] = extractor.extract(img_bgr)
    return feats


def run_retrieval(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """For every image as a query, return retrieved labels ordered by similarity.

    Returns:
        (N, N-1) matrix — for each query, the labels of the other N-1 items
        in order of decreasing cosine similarity. (We exclude the query itself.)
    """
    # All vectors are unit-norm, so cosine similarity = dot product
    n = features.shape[0]
    sim = features @ features.T  # (N, N)

    # Zero out the diagonal so a query doesn't retrieve itself
    np.fill_diagonal(sim, -np.inf)

    # Argsort descending — most similar first
    rankings = np.argsort(-sim, axis=1)[:, : n - 1]  # (N, N-1)

    # Map indices to labels
    retrieved_labels = labels[rankings]  # (N, N-1)
    return retrieved_labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="Path to dataset dir")
    p.add_argument("--variant", choices=["1k", "10k"], default="1k")
    p.add_argument(
        "--extractor",
        default="combo",
        choices=["histogram", "hog", "orb", "combo"],
    )
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    args = p.parse_args()

    print(f"Dataset:   {args.data}  (Corel-{args.variant})")
    print(f"Extractor: {args.extractor}")
    print()

    dataset = CorelDataset(args.data, variant=args.variant)
    print(f"Loaded {len(dataset)} images, {dataset.num_classes} classes\n")

    extractor = get_extractor(args.extractor)

    t0 = time.time()
    features = build_feature_matrix(dataset, extractor)
    t_extract = time.time() - t0
    print(f"\nFeature extraction: {t_extract:.1f}s ({t_extract/len(dataset)*1000:.1f} ms/image)")

    t0 = time.time()
    retrieved = run_retrieval(features, dataset.labels)
    t_retr = time.time() - t0
    print(f"Retrieval matrix:   {t_retr:.2f}s")

    # On Corel, every class has exactly 100 images, so each query has 99 relevant
    # items in the remaining N-1.
    metrics = evaluate_retrieval(
        retrieved,
        dataset.labels,
        k_values=tuple(args.k),
        total_relevant_per_class=99,
    )

    print("\n=== Results ===")
    print(format_metrics_table(metrics))


if __name__ == "__main__":
    main()
