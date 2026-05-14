"""FAISS vector index — build, save, load, search.

FAISS (Facebook AI Similarity Search) is the standard library for fast
nearest-neighbor search over millions of high-dimensional vectors.
It's used in production at Meta, Pinterest, Spotify, etc.

For Wang Corel (1K or 10K images) the dataset is small enough that a
brute-force exact search (IndexFlatIP) runs in microseconds — no need
for the fancier approximate indexes like IVF or HNSW. We use FlatIP
("inner product") because all our feature vectors are L2-normalized,
which means inner product == cosine similarity.

The index stores the vectors. We separately store a mapping from index
position → image path/label, since FAISS only knows about integer IDs.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class SearchHit:
    """One result from a search."""
    image_path: str
    label: int
    label_name: str
    score: float          # cosine similarity in [-1, 1], higher = more similar
    rank: int             # 1 = best match


class FAISSIndex:
    """A wrapper around a FAISS flat inner-product index.

    Tracks metadata (paths, labels, label names) alongside the vectors so
    that searches return useful results, not just integer IDs.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        # IndexFlatIP = exact search by inner product.
        # For unit-norm vectors, inner product = cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self.label_names: list[str] = []
        self.backend_name: str = "unknown"  # set by build script — used in the UI

    def add(
        self,
        vectors: np.ndarray,
        image_paths: list[str],
        labels: list[int],
        label_names: list[str],
    ) -> None:
        """Add a batch of vectors with their metadata."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of shape (N, {self.dim}), got {vectors.shape}"
            )
        if len(image_paths) != len(vectors):
            raise ValueError("image_paths length must match number of vectors")

        # FAISS expects float32, contiguous
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.image_paths.extend(image_paths)
        self.labels.extend(labels)
        self.label_names.extend(label_names)

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]:
        """Find the top-K most similar items to a query vector.

        Args:
            query: shape (dim,) or (1, dim) — must be L2-normalized already.
            k: number of results.
        """
        if query.ndim == 1:
            query = query[np.newaxis, :]
        query = np.ascontiguousarray(query, dtype=np.float32)

        # FAISS returns (scores, indices) — both shape (1, k)
        scores, indices = self.index.search(query, k)
        scores, indices = scores[0], indices[0]

        hits: list[SearchHit] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:  # FAISS uses -1 when fewer than k items exist
                continue
            hits.append(
                SearchHit(
                    image_path=self.image_paths[idx],
                    label=self.labels[idx],
                    label_name=self.label_names[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return hits

    @property
    def size(self) -> int:
        return self.index.ntotal

    # --- persistence ---
    def save(self, dir_path: str | Path) -> None:
        """Save index + metadata to disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        meta = {
            "dim": self.dim,
            "backend_name": self.backend_name,
            "image_paths": self.image_paths,
            "labels": self.labels,
            "label_names": self.label_names,
        }
        with open(dir_path / "meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, dir_path: str | Path) -> "FAISSIndex":
        dir_path = Path(dir_path)
        with open(dir_path / "meta.json") as f:
            meta = json.load(f)

        obj = cls(dim=meta["dim"])
        obj.index = faiss.read_index(str(dir_path / "index.faiss"))
        obj.image_paths = meta["image_paths"]
        obj.labels = meta["labels"]
        obj.label_names = meta["label_names"]
        obj.backend_name = meta.get("backend_name", "unknown")
        return obj


if __name__ == "__main__":
    # Smoke test: build a tiny index, search, verify.
    rng = np.random.default_rng(0)
    dim = 64
    n = 100
    # Random unit-norm vectors
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    idx = FAISSIndex(dim=dim)
    idx.add(
        vecs,
        image_paths=[f"img_{i}.jpg" for i in range(n)],
        labels=[i % 10 for i in range(n)],
        label_names=[f"class_{i % 10}" for i in range(n)],
    )
    # Query with the first vector — should match itself perfectly
    hits = idx.search(vecs[0], k=5)
    assert hits[0].image_path == "img_0.jpg"
    assert abs(hits[0].score - 1.0) < 1e-5
    print(f"Built index of {idx.size} vectors. Top-5 search:")
    for h in hits:
        print(f"  rank={h.rank} path={h.image_path} score={h.score:.4f}")
