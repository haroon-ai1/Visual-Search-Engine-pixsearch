"""Wang Corel dataset loader.

Corel-1K layout:
    data/corel1k/
        0.jpg, 1.jpg, ..., 999.jpg
    Class of image i = i // 100  (10 classes, 100 images each)

Corel-10K layout:
    data/corel10k/
        1.jpg, 2.jpg, ..., 10000.jpg
    Class of image i = (i - 1) // 100  (100 classes, 100 images each)

Ground-truth class names for Corel-1K (the canonical labels used in CBIR papers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


COREL1K_CLASSES = [
    "africa",      # 0
    "beach",       # 1
    "buildings",   # 2
    "buses",       # 3
    "dinosaurs",   # 4
    "elephants",   # 5
    "flowers",     # 6
    "horses",      # 7
    "mountains",   # 8
    "food",        # 9
]


@dataclass(frozen=True)
class ImageRecord:
    """A single image in the corpus."""
    image_id: int           # numeric id used in the filename
    path: Path              # absolute path on disk
    label: int              # class index
    label_name: str         # human-readable class

    def load(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Load the image as an RGB numpy array (H, W, 3), uint8."""
        img = Image.open(self.path).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        return np.array(img)


class CorelDataset:
    """Indexed access to a Wang Corel dataset.

    Use it like a list:
        ds = CorelDataset("data/corel1k", variant="1k")
        print(len(ds))           # 1000
        rec = ds[0]              # ImageRecord
        img = rec.load((224, 224))
    """

    def __init__(self, root: str | Path, variant: str = "1k") -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}\n"
                f"Run: python scripts/download_data.py --dataset corel{variant}"
            )
        if variant not in {"1k", "10k"}:
            raise ValueError(f"variant must be '1k' or '10k', got {variant!r}")

        self.variant = variant
        self.records: list[ImageRecord] = self._index_images()

        if len(self.records) == 0:
            raise RuntimeError(
                f"No images found in {self.root}. "
                f"Expected .jpg files named 0.jpg..N.jpg"
            )

    def _index_images(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        # Corel-1K: ids 0..999, class = id // 100
        # Corel-10K: ids 1..10000, class = (id - 1) // 100
        for path in sorted(self.root.glob("*.jpg"), key=self._sort_key):
            try:
                image_id = int(path.stem)
            except ValueError:
                continue  # ignore non-numeric filenames

            if self.variant == "1k":
                label = image_id // 100
                label_name = (
                    COREL1K_CLASSES[label]
                    if 0 <= label < len(COREL1K_CLASSES)
                    else f"class_{label}"
                )
            else:
                label = (image_id - 1) // 100
                label_name = f"class_{label:03d}"  # Corel-10K has no canonical names

            records.append(
                ImageRecord(image_id=image_id, path=path, label=label, label_name=label_name)
            )
        return records

    @staticmethod
    def _sort_key(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError:
            return -1

    # --- list-like API ---
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ImageRecord:
        return self.records[idx]

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self.records)

    # --- convenience ---
    @property
    def labels(self) -> np.ndarray:
        """Ground-truth labels for every record, in order."""
        return np.array([r.label for r in self.records], dtype=np.int64)

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1

    def class_indices(self, label: int) -> np.ndarray:
        """Indices (into self.records) of all images with the given label."""
        return np.where(self.labels == label)[0]


if __name__ == "__main__":
    # Smoke test
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/corel1k"
    variant = sys.argv[2] if len(sys.argv) > 2 else "1k"
    ds = CorelDataset(root, variant=variant)
    print(f"Loaded {len(ds)} images, {ds.num_classes} classes")
    print(f"First record: {ds[0]}")
    print(f"Class distribution: {np.bincount(ds.labels)}")
