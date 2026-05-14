"""Standard CBIR evaluation metrics.

Given a query and its retrieved ranked list of items, judge how good the ranking is.
An item is "relevant" if it shares the query's class label.

These are the metrics every CBIR paper reports — Precision@K, Recall@K, and
mean Average Precision (mAP). Implementing them yourself (rather than calling
sklearn) is the standard practice in retrieval because the definitions vary
slightly between papers and you want them transparent.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(retrieved_labels: np.ndarray, query_label: int, k: int) -> float:
    """Fraction of the top-K retrieved items that are relevant.

    Args:
        retrieved_labels: shape (N,) — labels of retrieved items, ordered by rank.
        query_label: the ground-truth class of the query.
        k: cutoff.

    Returns:
        P@K in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved_labels[:k]
    if len(top_k) == 0:
        return 0.0
    return float((top_k == query_label).sum() / k)


def recall_at_k(
    retrieved_labels: np.ndarray,
    query_label: int,
    k: int,
    total_relevant: int,
) -> float:
    """Fraction of all relevant items that appear in the top-K.

    Args:
        retrieved_labels: shape (N,) — labels of retrieved items, ordered by rank.
        query_label: the ground-truth class of the query.
        k: cutoff.
        total_relevant: total number of relevant items in the corpus
                        (for Corel-1K with the query excluded: 99).
    """
    if total_relevant <= 0:
        return 0.0
    top_k = retrieved_labels[:k]
    return float((top_k == query_label).sum() / total_relevant)


def average_precision(retrieved_labels: np.ndarray, query_label: int) -> float:
    """Average Precision for one query.

    AP = mean of P@k for every position k where the item at rank k is relevant.
    This is the canonical definition used by TREC and the CBIR literature.
    """
    relevant_mask = (retrieved_labels == query_label).astype(np.float32)
    if relevant_mask.sum() == 0:
        return 0.0
    # Cumulative count of hits up to each rank
    cum_hits = np.cumsum(relevant_mask)
    ranks = np.arange(1, len(relevant_mask) + 1, dtype=np.float32)
    precisions_at_hits = (cum_hits / ranks) * relevant_mask
    return float(precisions_at_hits.sum() / relevant_mask.sum())


def evaluate_retrieval(
    all_retrieved_labels: np.ndarray,
    all_query_labels: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    total_relevant_per_class: int | None = None,
) -> dict[str, float]:
    """Aggregate metrics over many queries.

    Args:
        all_retrieved_labels: shape (Q, N) — retrieved labels for each query.
        all_query_labels:     shape (Q,)   — ground-truth label for each query.
        k_values:             which K cutoffs to report.
        total_relevant_per_class: total relevant items per query (assumed
            constant — true for Corel where every class has 100 images).
            If None, computed from the data.

    Returns:
        Dict with keys 'P@k', 'R@k', and 'mAP'.
    """
    q, n = all_retrieved_labels.shape
    assert all_query_labels.shape == (q,), "query labels must match"

    if total_relevant_per_class is None:
        # Estimate from the data: for each query, count items of same class
        # in the full retrieval list.
        per_query_relevant = np.array(
            [(all_retrieved_labels[i] == all_query_labels[i]).sum() for i in range(q)]
        )
        total_relevant_per_class = int(per_query_relevant.max())

    results: dict[str, float] = {}

    for k in k_values:
        ps, rs = [], []
        for i in range(q):
            ps.append(precision_at_k(all_retrieved_labels[i], all_query_labels[i], k))
            rs.append(
                recall_at_k(
                    all_retrieved_labels[i],
                    all_query_labels[i],
                    k,
                    total_relevant_per_class,
                )
            )
        results[f"P@{k}"] = float(np.mean(ps))
        results[f"R@{k}"] = float(np.mean(rs))

    aps = [
        average_precision(all_retrieved_labels[i], all_query_labels[i])
        for i in range(q)
    ]
    results["mAP"] = float(np.mean(aps))

    return results


def format_metrics_table(metrics: dict[str, float]) -> str:
    """Pretty-print metrics as a single-line summary."""
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)
