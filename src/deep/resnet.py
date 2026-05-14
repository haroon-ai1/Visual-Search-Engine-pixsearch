"""ResNet-50 deep feature extractor.

Uses a pre-trained ResNet-50 (trained on ImageNet by the PyTorch team) as
a *frozen* feature extractor. We chop off the final classification layer
and use the 2048-dimensional pooled feature from the layer before it.

Why this works for retrieval:
  ImageNet has 1.2M images across 1000 categories. The penultimate layer
  has learned to map images into a 2048-d space where visually similar
  images cluster together. We don't need to fine-tune for retrieval to
  work — we just steal the pre-trained representation.

This is the same trick used by every "transfer learning" tutorial,
and it's also what real production systems do as a baseline before
moving to specialized models like CLIP or DINO.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


class ResNet50Embedder:
    """Wraps a pre-trained ResNet-50 as a 2048-d feature extractor."""

    name = "resnet50"
    dim = 2048

    def __init__(self, device: str | None = None) -> None:
        # Pick GPU if available, else CPU. Works either way.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained ResNet-50. The weights are downloaded once
        # (~100MB) and cached locally by torchvision.
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights)

        # Remove the final FC layer — keep everything up to the global
        # average pool, which gives us a 2048-d vector per image.
        # ResNet's structure: [conv1, bn1, relu, maxpool, layer1..4, avgpool, fc]
        # We keep all but the last (fc).
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.backbone.eval()
        self.backbone.to(self.device)

        # Freeze — we never train.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Standard ImageNet preprocessing — must match how the model
        # was trained, or features will be garbage.
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def extract_batch(self, image_paths: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Extract features for a list of image paths.

        Returns an (N, 2048) float32 array, L2-normalized so that
        cosine similarity reduces to a dot product.
        """
        feats = np.zeros((len(image_paths), self.dim), dtype=np.float32)

        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc="extract:resnet50",
        ):
            end = min(start + batch_size, len(image_paths))
            batch_paths = image_paths[start:end]

            # Load + preprocess every image in the batch
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self.preprocess(img))
            batch = torch.stack(tensors).to(self.device)

            # Forward pass — output shape (B, 2048, 1, 1) because of the
            # avgpool layer, so we flatten the trailing dims.
            out = self.backbone(batch).squeeze(-1).squeeze(-1)  # (B, 2048)
            out = out.cpu().numpy().astype(np.float32)

            # L2-normalize each row — required for cosine similarity
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            out = out / norms

            feats[start:end] = out

        return feats

    @torch.no_grad()
    def extract_one(self, image_path: str) -> np.ndarray:
        """Single-image convenience wrapper, returns shape (2048,)."""
        return self.extract_batch([image_path])[0]

    @torch.no_grad()
    def extract_from_pil(self, pil_image) -> np.ndarray:
        """Extract from an in-memory PIL image — for the Streamlit UI."""
        tensor = self.preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        out = self.backbone(tensor).squeeze(-1).squeeze(-1)  # (1, 2048)
        out = out.cpu().numpy().astype(np.float32)[0]
        norm = np.linalg.norm(out)
        if norm > 1e-10:
            out = out / norm
        return out


if __name__ == "__main__":
    # Smoke test
    import sys

    paths = sys.argv[1:]
    if not paths:
        print("Usage: python -m src.deep.resnet img1.jpg img2.jpg ...")
        sys.exit(1)

    embedder = ResNet50Embedder()
    print(f"Device: {embedder.device}")
    feats = embedder.extract_batch(paths)
    print(f"Features shape: {feats.shape}")
    print(f"Norms (should all be ~1.0): {np.linalg.norm(feats, axis=1)}")
