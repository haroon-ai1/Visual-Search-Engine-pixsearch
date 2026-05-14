"""Prepare PixSearch for HuggingFace deployment.

Run this in Colab AFTER building all three indexes.

What it does:
  1. Copies 10 images per class (100 total) into data/demo/
  2. Rebuilds all three indexes pointing at data/demo/ with RELATIVE paths
  3. Fixes any existing indexes that have absolute /content/... paths
  4. Saves everything to Google Drive as backup

After running this, your repo is ready to push to HuggingFace.

Usage (in Colab):
    !PYTHONPATH=. python scripts/prepare_deploy.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("/content/pixsearch")
COREL_DIR     = PROJECT_ROOT / "data" / "corel1k"
DEMO_DIR      = PROJECT_ROOT / "data" / "demo"
INDEXES_DIR   = PROJECT_ROOT / "indexes"
DRIVE_BACKUP  = Path("/content/drive/MyDrive/pixsearch_deploy")

IMAGES_PER_CLASS = 10   # 10 × 10 classes = 100 images total, ~8 MB
# ──────────────────────────────────────────────────────────────────────────


def step1_create_demo_subset() -> None:
    print("\n" + "="*60)
    print("STEP 1 — Creating demo subset (100 images)")
    print("="*60)

    if not COREL_DIR.exists():
        print(f"❌  {COREL_DIR} not found.")
        print("    Make sure you're running this after uploading the dataset.")
        sys.exit(1)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for cls in range(10):
        for i in range(IMAGES_PER_CLASS):
            img_id = cls * 100 + i
            src = COREL_DIR / f"{img_id}.jpg"
            dst = DEMO_DIR / f"{img_id}.jpg"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1

    print(f"✓  Copied {copied} images to {DEMO_DIR}")


def step2_build_demo_indexes() -> None:
    print("\n" + "="*60)
    print("STEP 2 — Building demo indexes with relative paths")
    print("="*60)

    # We build from inside the project root so paths come out relative
    os.chdir(PROJECT_ROOT)
    env = {**os.environ, "PYTHONPATH": "."}

    backends = [
        ("classical", "indexes/classical_demo"),
        ("resnet",    "indexes/resnet_demo"),
        ("clip",      "indexes/clip_demo"),
    ]

    for backend, out in backends:
        print(f"\n  Building {backend}...")
        result = subprocess.run(
            [
                sys.executable, "scripts/build_index.py",
                "--data", "data/demo",
                "--variant", "1k",
                "--backend", backend,
                "--out", out,
                "--batch-size", "64",
            ],
            env=env,
        )
        if result.returncode != 0:
            print(f"  ❌  {backend} index build failed.")
            sys.exit(1)
        print(f"  ✓  {out} saved")


def step3_verify_paths() -> None:
    print("\n" + "="*60)
    print("STEP 3 — Verifying paths are relative")
    print("="*60)

    for meta_file in INDEXES_DIR.glob("*_demo/meta.json"):
        with open(meta_file) as f:
            meta = json.load(f)
        sample = meta["image_paths"][0]
        if sample.startswith("/"):
            print(f"  ❌  {meta_file.parent.name}: paths are still absolute!")
            print(f"      Sample: {sample}")
            sys.exit(1)
        else:
            print(f"  ✓  {meta_file.parent.name}: paths are relative ({sample})")


def step4_backup_to_drive() -> None:
    print("\n" + "="*60)
    print("STEP 4 — Backing up to Google Drive")
    print("="*60)

    drive_root = Path("/content/drive")
    if not drive_root.exists():
        print("  ⚠  Google Drive not mounted. Skipping backup.")
        print("     Mount with: from google.colab import drive; drive.mount('/content/drive')")
        return

    DRIVE_BACKUP.mkdir(parents=True, exist_ok=True)

    # Backup demo indexes
    idx_dst = DRIVE_BACKUP / "indexes"
    if idx_dst.exists():
        shutil.rmtree(idx_dst)
    shutil.copytree(INDEXES_DIR, idx_dst)
    print(f"  ✓  Indexes backed up to {idx_dst}")

    # Backup demo images
    img_dst = DRIVE_BACKUP / "data" / "demo"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    if img_dst.exists():
        shutil.rmtree(img_dst)
    shutil.copytree(DEMO_DIR, img_dst)
    print(f"  ✓  Demo images backed up to {img_dst}")

    print(f"\n  Drive backup complete: {DRIVE_BACKUP}")


def step5_print_summary() -> None:
    print("\n" + "="*60)
    print("STEP 5 — Summary")
    print("="*60)

    total_size = 0
    files_to_push = []

    # Count what needs to go into the HF repo
    for path in INDEXES_DIR.glob("*_demo/**/*"):
        if path.is_file():
            total_size += path.stat().st_size
            files_to_push.append(str(path.relative_to(PROJECT_ROOT)))

    for path in DEMO_DIR.glob("*.jpg"):
        total_size += path.stat().st_size
        files_to_push.append(str(path.relative_to(PROJECT_ROOT)))

    print(f"\n  Files ready for HuggingFace repo:")
    for f in sorted(files_to_push)[:10]:
        print(f"    {f}")
    if len(files_to_push) > 10:
        print(f"    ... and {len(files_to_push)-10} more")

    print(f"\n  Total size to push: {total_size / 1024 / 1024:.1f} MB")
    print(f"\n  ✅  Ready to deploy! Follow DEPLOY.md for next steps.")


if __name__ == "__main__":
    step1_create_demo_subset()
    step2_build_demo_indexes()
    step3_verify_paths()
    step4_backup_to_drive()
    step5_print_summary()
