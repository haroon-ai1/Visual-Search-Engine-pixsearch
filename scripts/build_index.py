"""Build a FAISS index over a Wang Corel dataset.

Usage:
    PYTHONPATH=. python scripts/build_index.py \
        --data data/corel1k --variant 1k --backend classical --out indexes/classical_1k

    PYTHONPATH=. python scripts/build_index.py \
        --data data/corel1k --variant 1k --backend resnet --out indexes/resnet_1k

    PYTHONPATH=. python scripts/build_index.py \
        --data data/corel1k --variant 1k --backend clip --out indexes/clip_1k

The index + metadata is saved to --out as two files (index.faiss, meta.json)
which the Streamlit app loads at startup.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.classical.extractors import get_extractor
from src.index.faiss_index import FAISSIndex
from src.utils.dataset import CorelDataset


def build_classical(dataset: CorelDataset, extractor_name: str) -> tuple[np.ndarray, str]:
    extractor = get_extractor(extractor_name)
    feats = np.zeros((len(dataset), extractor.dim), dtype=np.float32)
    for i, rec in enumerate(tqdm(dataset, desc=f"extract:{extractor.name}")):
        img = cv2.imread(str(rec.path))
        if img is None:
            raise RuntimeError(f"Failed to read {rec.path}")
        feats[i] = extractor.extract(img)
    return feats, f"classical_{extractor_name}"


def build_resnet(dataset: CorelDataset, batch_size: int) -> tuple[np.ndarray, str]:
    from src.deep.resnet import ResNet50Embedder
    embedder = ResNet50Embedder()
    print(f"ResNet-50 device: {embedder.device}")
    paths = [str(rec.path) for rec in dataset]
    feats = embedder.extract_batch(paths, batch_size=batch_size)
    return feats, "resnet50"


def build_clip(dataset: CorelDataset, batch_size: int) -> tuple[np.ndarray, str]:
    """Extract CLIP image features.

    The index stores image vectors (512-d). At search time, the Streamlit app
    can query with EITHER an uploaded image OR a text string — both are embedded
    into the same 512-d space by the CLIPEmbedder, so one index handles both.
    """
    from src.deep.clip_embedder import CLIPEmbedder
    embedder = CLIPEmbedder()
    paths = [str(rec.path) for rec in dataset]
    feats = embedder.embed_images(paths, batch_size=batch_size)
    return feats, "clip"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--variant", choices=["1k", "10k"], default="1k")
    p.add_argument(
        "--backend",
        choices=["histogram", "hog", "orb", "classical", "resnet", "clip"],
        default="resnet",
        help="'classical'=combo histogram+hog+orb, 'resnet'=ResNet-50, 'clip'=CLIP ViT-B/32",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    print(f"Dataset:  {args.data}  (Corel-{args.variant})")
    print(f"Backend:  {args.backend}")
    print(f"Output:   {args.out}\n")

    dataset = CorelDataset(args.data, variant=args.variant)
    print(f"Loaded {len(dataset)} images, {dataset.num_classes} classes\n")

    t0 = time.time()
    if args.backend == "resnet":
        features, backend_name = build_resnet(dataset, args.batch_size)
    elif args.backend == "clip":
        features, backend_name = build_clip(dataset, args.batch_size)
    else:
        ext_name = "combo" if args.backend == "classical" else args.backend
        features, backend_name = build_classical(dataset, ext_name)

    elapsed = time.time() - t0
    print(f"\nFeature extraction: {elapsed:.1f}s  ({elapsed/len(dataset)*1000:.1f} ms/image)")
    print(f"Feature matrix: {features.shape}, dtype={features.dtype}")

    idx = FAISSIndex(dim=features.shape[1])
    idx.backend_name = backend_name
    idx.add(
        features,
        image_paths=[str(r.path) for r in dataset],
        labels=[int(r.label) for r in dataset],
        label_names=[r.label_name for r in dataset],
    )
    idx.save(args.out)
    print(f"\nSaved index ({idx.size} vectors) to {args.out}")


if __name__ == "__main__":
    main()
