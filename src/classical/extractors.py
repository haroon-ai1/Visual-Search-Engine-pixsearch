"""Classical computer vision feature extractors.

These three are the foundation of pre-deep-learning CBIR and they map directly
to the AICL 3605 course outline:

  - Color histogram         → Wk 3-4 (point processing, scene enhancement)
  - HOG (Histogram of       → Wk 5 (edge detection), Wk 7 (feature extraction)
    Oriented Gradients)
  - ORB (Oriented FAST +    → Wk 6-7 (feature detection, registration)
    Rotated BRIEF)

Each extractor returns a fixed-length feature vector per image. The vectors
from a corpus are L2-normalized so cosine similarity reduces to a dot product
(which is what FAISS expects).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from skimage.feature import hog


class FeatureExtractor(ABC):
    """All extractors return an L2-normalized 1-D feature vector."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        """Extract a 1-D float32 feature vector from a BGR uint8 image."""

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v.astype(np.float32)
        return (v / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Color histogram (HSV, 3D joint histogram)
# ---------------------------------------------------------------------------

class ColorHistogram(FeatureExtractor):
    """3D HSV joint histogram.

    HSV is preferred over RGB for retrieval because hue is roughly invariant
    to lighting changes. We use a coarse 8x8x8 = 512-bin histogram, which is
    a very standard choice in the CBIR literature.
    """

    name = "color_histogram"

    def __init__(self, bins: tuple[int, int, int] = (8, 8, 8)) -> None:
        self.bins = bins
        self.dim = bins[0] * bins[1] * bins[2]

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # H: 0-180 in OpenCV, S: 0-256, V: 0-256
        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=list(self.bins),
            ranges=[0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        return self._l2_normalize(hist)


# ---------------------------------------------------------------------------
# 2. HOG (Histogram of Oriented Gradients)
# ---------------------------------------------------------------------------

class HOGExtractor(FeatureExtractor):
    """HOG over a resized grayscale image.

    HOG captures local edge / shape structure — complements color histograms
    nicely because color says nothing about geometry.
    """

    name = "hog"

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (16, 16),
        cells_per_block: tuple[int, int] = (2, 2),
    ) -> None:
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        # Pre-compute output dim (skimage convention)
        cells_x = image_size[0] // pixels_per_cell[0]
        cells_y = image_size[1] // pixels_per_cell[1]
        blocks_x = cells_x - cells_per_block[0] + 1
        blocks_y = cells_y - cells_per_block[1] + 1
        self.dim = (
            blocks_x * blocks_y
            * cells_per_block[0] * cells_per_block[1]
            * orientations
        )

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.image_size)
        feat = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
        return self._l2_normalize(feat)


# ---------------------------------------------------------------------------
# 3. ORB — bag of visual words style, fixed length
# ---------------------------------------------------------------------------

class ORBHistogram(FeatureExtractor):
    """ORB keypoint descriptors aggregated into a fixed-size signature.

    Raw ORB gives a variable number of 256-bit descriptors per image, which
    can't be directly compared with cosine similarity. We aggregate by taking
    the bit-frequency histogram: for each of the 256 bits, how often is it
    set across the keypoints in this image? That gives a fixed 256-d vector.

    This is a simplified bag-of-words: cheap to compute, no clustering
    needed, and works well enough to demonstrate the technique. Real-world
    systems would train a visual vocabulary with k-means.
    """

    name = "orb"
    dim = 256

    def __init__(self, n_features: int = 500) -> None:
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            # No keypoints — return a zero vector (will be L2-normalized to zero,
            # which is fine, just means very low similarity to everything)
            return np.zeros(self.dim, dtype=np.float32)

        # descriptors: (K, 32) uint8. Unpack each byte into 8 bits → (K, 256) uint8.
        bits = np.unpackbits(descriptors, axis=1).astype(np.float32)  # (K, 256)
        signature = bits.mean(axis=0)  # (256,)
        return self._l2_normalize(signature)


# ---------------------------------------------------------------------------
# Concatenated combo extractor
# ---------------------------------------------------------------------------

class CombinedClassical(FeatureExtractor):
    """Concatenate histogram + HOG + ORB features.

    Each component is L2-normalized first (so it lives on the unit sphere),
    then they're concatenated and the result is L2-normalized again. This
    weights all three components equally; in practice you'd tune weights.
    """

    name = "classical_combo"

    def __init__(self) -> None:
        self.histogram = ColorHistogram()
        self.hog_ext = HOGExtractor()
        self.orb_ext = ORBHistogram()
        self.dim = self.histogram.dim + self.hog_ext.dim + self.orb_ext.dim

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        h = self.histogram.extract(image_bgr)
        g = self.hog_ext.extract(image_bgr)
        o = self.orb_ext.extract(image_bgr)
        combined = np.concatenate([h, g, o])
        return self._l2_normalize(combined)


# ---------------------------------------------------------------------------
# Registry — easy lookup by name
# ---------------------------------------------------------------------------

EXTRACTORS: dict[str, type[FeatureExtractor]] = {
    "histogram": ColorHistogram,
    "hog": HOGExtractor,
    "orb": ORBHistogram,
    "combo": CombinedClassical,
}


def get_extractor(name: str) -> FeatureExtractor:
    if name not in EXTRACTORS:
        raise KeyError(
            f"Unknown classical extractor {name!r}. "
            f"Available: {list(EXTRACTORS.keys())}"
        )
    return EXTRACTORS[name]()


if __name__ == "__main__":
    # Smoke test on a synthetic image
    img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    for name in EXTRACTORS:
        ext = get_extractor(name)
        feat = ext.extract(img)
        print(f"{name:12s} -> shape={feat.shape}, dim={ext.dim}, "
              f"||v||={np.linalg.norm(feat):.4f}")
