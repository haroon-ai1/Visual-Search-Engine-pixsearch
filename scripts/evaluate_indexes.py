"""Evaluate one or more FAISS indexes and print a comparison table.

Usage:
    PYTHONPATH=. python scripts/evaluate_indexes.py \\
        indexes/classical_1k indexes/resnet_1k

This is the script that produces the centerpiece numbers for your README
and class presentation: classical mAP vs deep mAP, side by side.

It works by leave-one-out evaluation: for every image in the index, use
it as a query, retrieve the top-K most similar items from the rest of
the index, and compute retrieval metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.index.faiss_index import FAISSIndex
from src.utils.metrics import evaluate_retrieval, format_metrics_table


def evaluate_index(index_dir: Path, k_values: tuple[int, ...]) -> dict[str, float]:
    """Run leave-one-out evaluation over the full index."""
    idx = FAISSIndex.load(index_dir)
    n = idx.size

    # Get all vectors back from the FAISS index (we need them as queries).
    # IndexFlatIP stores them as-is, so we can pull them out cheaply.
    import faiss
    all_vecs = idx.index.reconstruct_n(0, n)  # (N, dim)

    labels = np.array(idx.labels)
    # For each query, we ask for n results (the query will be in there with
    # score ~1.0, we'll drop it). FAISS doesn't have a built-in "exclude self"
    # so we ask for n and skip the self-match.
    all_retrieved = np.zeros((n, n - 1), dtype=np.int64)

    for i in tqdm(range(n), desc=f"eval:{idx.backend_name}"):
        hits = idx.search(all_vecs[i], k=n)
        # Drop the self-match (whichever hit corresponds to this image's path)
        own_path = idx.image_paths[i]
        retrieved_labels = []
        for h in hits:
            if h.image_path == own_path:
                continue
            retrieved_labels.append(h.label)
            if len(retrieved_labels) == n - 1:
                break
        all_retrieved[i] = retrieved_labels

    # Corel always has 100 images per class → 99 relevant per query
    return evaluate_retrieval(
        all_retrieved,
        labels,
        k_values=k_values,
        total_relevant_per_class=99,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("indexes", type=Path, nargs="+", help="Index directories to evaluate")
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    args = p.parse_args()

    results: dict[str, dict[str, float]] = {}
    for idx_dir in args.indexes:
        print(f"\n--- Evaluating {idx_dir} ---")
        m = evaluate_index(idx_dir, tuple(args.k))
        results[idx_dir.name] = m
        print(f"  {format_metrics_table(m)}")

    # Side-by-side table for README / slides
    print("\n\n=== Comparison Table ===")
    metric_keys = list(next(iter(results.values())).keys())
    header = "Backend".ljust(25) + "".join(f"{k:>10}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = name.ljust(25) + "".join(f"{m[k]:>10.4f}" for k in metric_keys)
        print(row)


if __name__ == "__main__":
    main()
