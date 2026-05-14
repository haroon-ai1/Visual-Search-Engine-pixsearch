"""CLIP feature extractor — image AND text queries in the same vector space.

CLIP (Contrastive Language-Image Pretraining) was published by OpenAI in 2021.
The key insight: train a model on 400 million (image, caption) pairs from the
internet so that matching image-text pairs end up close together in a shared
512-dimensional embedding space.

What this means for retrieval:
  - You can embed a TEXT query like "a horse running in a field" and search
    for visually similar images — without any text labels in your dataset.
  - You can embed an IMAGE and search with that too — same as ResNet.
  - Both live in the same 512-d space, so one FAISS index handles both.

This is what makes CLIP the "killer move" for demo purposes — text-to-image
search impresses an audience in a way that image-to-image can't.

Paper:
  Radford et al., "Learning Transferable Visual Models From Natural Language
  Supervision", ICML 2021. https://arxiv.org/abs/2103.00020

Model used here:
  ViT-B/32 — the smallest CLIP model (~350 MB). Fast enough for a demo,
  good enough for Corel-1K. Larger models (ViT-L/14) give better results
  but are slower to download and run.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


class CLIPEmbedder:
    """Wraps OpenAI CLIP ViT-B/32 for both image and text embedding."""

    name = "clip"
    dim = 512   # ViT-B/32 output dimension

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model, preprocessing pipeline, and tokenizer.
        # open_clip downloads weights on first call (~350 MB), cached after.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()
        self.model.to(self.device)

        # Freeze — we never train CLIP
        for p in self.model.parameters():
            p.requires_grad = False

        print(f"CLIP ({model_name}/{pretrained}) loaded on {self.device}")

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return v / norms

    # ------------------------------------------------------------------
    # Image embedding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_images(
        self, image_paths: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        """Embed a list of image paths → (N, 512) float32, L2-normalized."""
        feats = np.zeros((len(image_paths), self.dim), dtype=np.float32)

        for start in tqdm(
            range(0, len(image_paths), batch_size), desc="extract:clip_image"
        ):
            end = min(start + batch_size, len(image_paths))
            tensors = []
            for p in image_paths[start:end]:
                img = Image.open(p).convert("RGB")
                tensors.append(self.preprocess(img))

            batch = torch.stack(tensors).to(self.device)
            # encode_image returns (B, 512) float32
            out = self.model.encode_image(batch).cpu().numpy().astype(np.float32)
            feats[start:end] = self._l2_normalize(out)

        return feats

    @torch.no_grad()
    def embed_image_pil(self, pil_image: Image.Image) -> np.ndarray:
        """Embed a single PIL image → shape (512,), L2-normalized."""
        tensor = self.preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        out = self.model.encode_image(tensor).cpu().numpy().astype(np.float32)[0]
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-10 else out

    # ------------------------------------------------------------------
    # Text embedding  ← the unique capability ResNet doesn't have
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a natural-language string → shape (512,), L2-normalized.

        The text is tokenized, passed through CLIP's text encoder, and
        projected into the same 512-d space as images. This is what allows
        queries like "a dinosaur on a white background" to retrieve
        actual dinosaur photos.
        """
        tokens = self.tokenizer([text]).to(self.device)   # (1, 77)
        out = self.model.encode_text(tokens).cpu().numpy().astype(np.float32)[0]
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-10 else out

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts → (N, 512), L2-normalized."""
        tokens = self.tokenizer(texts).to(self.device)
        out = self.model.encode_text(tokens).cpu().numpy().astype(np.float32)
        return self._l2_normalize(out)


if __name__ == "__main__":
    # Smoke test — no images needed
    embedder = CLIPEmbedder()

    texts = [
        "a horse running in a field",
        "an elephant in the savanna",
        "a beautiful beach at sunset",
        "a red double-decker bus",
        "colorful flowers in a garden",
    ]

    print("\nText embeddings:")
    vecs = embedder.embed_texts(texts)
    print(f"Shape: {vecs.shape}")
    print(f"Norms (should be ~1.0): {np.linalg.norm(vecs, axis=1).round(4)}")

    # Cross-similarity — "horse" should be closer to "horse" than to "bus"
    sim = vecs @ vecs.T
    print(f"\nCross-similarity (diagonal = 1.0):")
    for i, t in enumerate(texts):
        print(f"  {t[:35]:35s} → most similar: {texts[sim[i].argsort()[::-1][1]]}")
