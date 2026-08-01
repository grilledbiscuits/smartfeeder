"""One-off frozen-backbone feature extraction to .npy.

This is what makes the project tractable on a CPU-only laptop. Run the corpus
through the backbone once, save the penultimate-layer features, and every
subsequent head experiment becomes a matrix operation measured in seconds
instead of a forward pass measured in tens of minutes.

Deliberately architecture-agnostic. The same script runs locally against a small
CNN and on Kaggle against the ViT-L iNat-2021 teacher; the resulting .npy is
~80MB for 20k images at 1024 dims, small enough to download and feed straight
into the local fast loop. That is how the fast loop gets teacher-grade features
without a local GPU.

Features and labels are saved together and alignment is asserted on load. A
silent misalignment between a feature matrix and a label array produces
plausible-looking metrics that are pure noise.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from birdcam.config import Config, load_config
from birdcam.data.dataset import ImageDataset, load_labelled
from birdcam.data.manifest import open_manifest
from birdcam.models.backbone import build_transform, feature_dim, load_backbone
from birdcam.utils.runtime import setup_torch

logger = logging.getLogger(__name__)


def cache(cfg: Config, role: str = "local", limit: int | None = None) -> Path:
    import torch

    setup_torch(cfg)

    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
    if limit:
        items = items[:limit]
    if not items:
        raise RuntimeError("No labelled images found. Run the fetcher and preprocess first.")

    missing = [i for i in items if not i.path.is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} processed images are missing from disk "
            f"(e.g. {missing[0].path}). Re-run preprocess."
        )

    model, name = load_backbone(cfg, role, num_classes=0)
    transform = build_transform(cfg, model, role)
    dim = feature_dim(model)
    logger.info("extracting %d features/image from %d images using %s", dim, len(items), name)

    ds = ImageDataset(items, transform)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.train_cfg["backbone"][role]["batch_size"],
        shuffle=False,
        num_workers=cfg.train_cfg["compute"]["dataloader_num_workers"],
        pin_memory=cfg.train_cfg["compute"]["pin_memory"],
    )

    feats = np.zeros((len(items), dim), dtype=np.float32)
    t0 = time.monotonic()
    done = 0
    with torch.inference_mode():
        for batch, idx in loader:
            out = model(batch)
            feats[idx.numpy()] = out.float().numpy()
            done += len(idx)
            if done % 200 < len(idx):
                rate = done / max(time.monotonic() - t0, 1e-9)
                eta = (len(items) - done) / max(rate, 1e-9)
                logger.info("  %d/%d  %.1f img/s  eta %.0fs", done, len(items), rate, eta)

    elapsed = time.monotonic() - t0
    out_dir = cfg.path("embeddings_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{role}_{name.replace('/', '_')}"
    np.save(out_dir / f"{stem}.npy", feats)

    meta = {
        "backbone": name,
        "role": role,
        "feature_dim": dim,
        "n_images": len(items),
        "seconds": round(elapsed, 1),
        "images_per_second": round(len(items) / max(elapsed, 1e-9), 2),
        "items": [
            {
                "image_id": i.image_id,
                "scientific_name": i.scientific_name,
                "taxon_label": i.taxon_label,
                "taxon_index": i.taxon_index,
                "sex_label": i.sex_label_name,
                "sex_mask": i.sex_mask.tolist(),
                "split": i.split,
                "observation_id": i.observation_id,
                "observer_id": i.observer_id,
            }
            for i in items
        ],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
    logger.info(
        "wrote %s (%.1f MB) in %.0fs (%.1f img/s)",
        out_dir / f"{stem}.npy",
        feats.nbytes / 1e6,
        elapsed,
        len(items) / max(elapsed, 1e-9),
    )
    return out_dir / f"{stem}.npy"


def load_cached(cfg: Config, stem: str) -> tuple[np.ndarray, list[dict]]:
    """Load features + aligned metadata, asserting they match."""
    d = cfg.path("embeddings_dir")
    feats = np.load(d / f"{stem}.npy")
    meta = json.loads((d / f"{stem}.json").read_text(encoding="utf-8"))
    if len(feats) != len(meta["items"]):
        raise RuntimeError(
            f"feature/label misalignment: {len(feats)} rows vs {len(meta['items'])} items"
        )
    return feats, meta["items"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    import argparse

    ap = argparse.ArgumentParser(description="Cache frozen-backbone embeddings.")
    ap.add_argument("--role", default="local", choices=["local", "teacher", "student"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cache(load_config(), role=args.role, limit=args.limit)


if __name__ == "__main__":
    main()
