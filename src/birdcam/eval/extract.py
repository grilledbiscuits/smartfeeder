"""Extract pooled features and both heads' logits from a trained checkpoint.

Why this module exists
----------------------
Until 2026-08-05 everything in `eval/` loaded frozen backbone embeddings from
the sweep cache (`embeddings_dir/sweep/*.npy`) and never read a checkpoint. That
was correct while the only trained thing was a linear probe over those exact
embeddings. It stopped being correct the moment `train_full.py` fine-tuned the
backbone: re-running the evaluation after a fine-tune would silently refit a new
probe on the OLD features and report it as the new model. Temperature,
thresholds, novelty scores and quantisation deltas would all describe a model
nobody was going to deploy.

This module is the single source of model outputs for everything downstream.

What it emits
-------------
For every image, three aligned arrays:

* `features`     -- pooled backbone output, pre-dropout, pre-head. The kNN
                    novelty scorer needs these and nothing else provides them.
* `taxon_logits` -- raw, untempered. Calibration is a downstream decision.
* `sex_logits`   -- likewise.

Staleness
---------
The sweep cache was validated by row count alone, which cannot detect a manifest
that gained or lost images, a re-split, or a different checkpoint. Every file
written here carries a fingerprint over the checkpoint bytes, the ordered image
IDs and the taxon class order. `load_extraction` refuses a file whose
fingerprint no longer matches rather than returning plausible wrong numbers.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PHASE = 7


@dataclass
class Extraction:
    """Model outputs for one image set, aligned row-for-row with `image_ids`."""

    image_ids: np.ndarray  # (N,) str
    features: np.ndarray  # (N, D) float32
    taxon_logits: np.ndarray  # (N, C_taxon) float32
    sex_logits: np.ndarray  # (N, C_sex) float32
    taxon_classes: list[str]
    sex_classes: list[str]
    checkpoint: str
    checkpoint_sha: str
    fingerprint: str

    def __len__(self) -> int:
        return len(self.image_ids)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fingerprint(checkpoint_sha: str, image_ids, taxon_classes) -> str:
    """Content hash over everything that would invalidate the arrays.

    Ordered image IDs, not just their count: a manifest that swapped one image
    for another keeps the same length and would otherwise pass unnoticed.
    """
    h = hashlib.sha256()
    h.update(checkpoint_sha.encode())
    h.update(b"\x00")
    for i in image_ids:
        h.update(str(i).encode())
        h.update(b"\x1f")
    h.update(b"\x00")
    for c in taxon_classes:
        h.update(c.encode())
        h.update(b"\x1f")
    return h.hexdigest()


def load_model_from_checkpoint(cfg, checkpoint: Path, role: str = "student"):
    """Build the two-head model and load trained weights into it.

    Raises on a missing checkpoint rather than falling back to pretrained
    weights. A silent fallback here reproduces the exact bug this module was
    written to prevent, one layer down.
    """
    import torch

    from birdcam.models.heads import build_model

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint}. Extraction must run against trained "
            "weights; refusing to fall back to the pretrained backbone."
        )

    model, name = build_model(cfg, role)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"{checkpoint} has no 'model' key; not a train_full.py checkpoint")

    # strict=True: a checkpoint whose head width disagrees with the current
    # taxonomy must fail loudly, not load 62 classes into 22 slots.
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    # RunState.epoch is the *next* epoch to run, so the weights are from
    # epoch-1. best_val is the Tier A macro-recall that selected this file.
    st = payload.get("state", {})
    hist = st.get("history") or [{}]
    logger.info(
        "loaded %s from %s: %d epochs trained, selected Tier A recall %.4f, "
        "val acc %.4f at that epoch",
        name,
        checkpoint.name,
        st.get("epoch", 0),
        st.get("best_val", float("nan")),
        hist[-1].get("val_taxon_acc", float("nan")),
    )
    return model, name


def _forward_features_and_logits(model, batch):
    """Pooled features plus both heads, in one pass.

    `TwoHeadNet.forward` returns logits only, so the backbone is called
    directly. Dropout is identity in eval mode, so this is numerically the same
    path as `forward` -- it just also keeps the intermediate.
    """
    feats = model.backbone(batch)
    return feats, model.taxon_head(feats), model.sex_head(feats)


def extract(cfg, model, items, batch_size: int = 32, workers: int | None = None):
    """Run `items` through `model`, returning (features, taxon_logits, sex_logits).

    `items` may be any objects exposing `.path`; ordering of the returned rows
    matches `items` exactly.
    """
    import torch

    from birdcam.data.dataset import ImageDataset
    from birdcam.train.augment import build_eval_transform

    size = cfg.train_cfg["backbone"]["student"]["image_size"]
    transform = build_eval_transform(size)
    if workers is None:
        workers = cfg.train_cfg["compute"]["dataloader_num_workers"]

    loader = torch.utils.data.DataLoader(
        ImageDataset(items, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    feats = logits_t = logits_s = None
    done = 0
    with torch.inference_mode():
        for batch, idx in loader:
            f, lt, ls = _forward_features_and_logits(model, batch)
            if feats is None:
                feats = np.zeros((len(items), f.shape[1]), dtype=np.float32)
                logits_t = np.zeros((len(items), lt.shape[1]), dtype=np.float32)
                logits_s = np.zeros((len(items), ls.shape[1]), dtype=np.float32)
            j = idx.numpy()
            feats[j] = f.float().numpy()
            logits_t[j] = lt.float().numpy()
            logits_s[j] = ls.float().numpy()
            done += len(j)
            if done % (batch_size * 20) < batch_size:
                logger.info("  %d/%d images", done, len(items))

    return feats, logits_t, logits_s


def ood_items(cfg, m):
    """OOD rows as path-bearing stand-ins for LabelledImage.

    OOD images carry no split and no taxon label -- `preprocess.py` clears their
    split precisely so they cannot leak into training -- so `load_labelled`
    skips them. They still need features for the novelty scorer.
    """

    @dataclass
    class _OODItem:
        image_id: str
        path: Path
        scientific_name: str
        observation_id: str | None

    out = []
    for r in m.iter_rows("tier='OOD' AND status='downloaded'"):
        slug = r["scientific_name"].lower().replace(" ", "_")
        out.append(
            _OODItem(
                image_id=r["image_id"],
                path=cfg.path("processed_dir") / slug / f"{r['image_id'].replace(':', '_')}.jpg",
                scientific_name=r["scientific_name"],
                observation_id=r["observation_id"],
            )
        )
    return out


def run(cfg, checkpoint_name: str = "student_best.pt", batch_size: int = 32) -> dict[str, Path]:
    """Extract for both the labelled corpus and the OOD set; write two .npz files."""
    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest

    ckpt = cfg.path("checkpoints_dir") / checkpoint_name
    model, _ = load_model_from_checkpoint(cfg, ckpt)
    ckpt_sha = _sha256_file(ckpt)

    out_dir = cfg.path("embeddings_dir") / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_name.replace(".pt", "")
    written: dict[str, Path] = {}

    with open_manifest(cfg.path("manifest_db")) as m:
        sets = {"id": load_labelled(cfg, m), "ood": ood_items(cfg, m)}

    for kind, items in sets.items():
        if not items:
            logger.warning("no %s images found; skipping", kind)
            continue
        logger.info("extracting %s: %d images", kind, len(items))
        feats, lt, ls = extract(cfg, model, items, batch_size=batch_size)
        ids = np.array([i.image_id for i in items], dtype=object)
        path = out_dir / f"{stem}_{kind}.npz"
        np.savez_compressed(
            path,
            image_ids=ids,
            features=feats,
            taxon_logits=lt,
            sex_logits=ls,
            taxon_classes=np.array(cfg.taxon_classes, dtype=object),
            sex_classes=np.array(cfg.sex_classes, dtype=object),
            checkpoint=checkpoint_name,
            checkpoint_sha=ckpt_sha,
            fingerprint=_fingerprint(ckpt_sha, ids, cfg.taxon_classes),
        )
        logger.info("wrote %s  (%d x %d features)", path, feats.shape[0], feats.shape[1])
        written[kind] = path

    return written


def load_extraction(cfg, path: Path, expect_image_ids=None) -> Extraction:
    """Load an extraction, refusing it if the fingerprint no longer matches.

    A stale features file is worse than a missing one: it produces numbers that
    look right. If this raises, re-run `extract`.
    """
    z = np.load(path, allow_pickle=True)
    ids = z["image_ids"]
    taxon_classes = [str(c) for c in z["taxon_classes"]]
    stored = str(z["fingerprint"])
    recomputed = _fingerprint(str(z["checkpoint_sha"]), ids, taxon_classes)
    if stored != recomputed:
        raise RuntimeError(f"{path} is internally inconsistent; re-run extraction")

    if taxon_classes != list(cfg.taxon_classes):
        raise RuntimeError(
            f"{path} was written with a different taxon class order "
            f"({len(taxon_classes)} classes vs {len(cfg.taxon_classes)} now). Re-run extraction."
        )
    if expect_image_ids is not None:
        want = np.array([str(i) for i in expect_image_ids], dtype=object)
        if len(want) != len(ids) or not np.array_equal(want, ids.astype(object)):
            raise RuntimeError(
                f"{path} does not match the current manifest "
                f"({len(ids)} rows stored vs {len(want)} expected). Re-run extraction."
            )

    return Extraction(
        image_ids=ids,
        features=z["features"],
        taxon_logits=z["taxon_logits"],
        sex_logits=z["sex_logits"],
        taxon_classes=taxon_classes,
        sex_classes=[str(c) for c in z["sex_classes"]],
        checkpoint=str(z["checkpoint"]),
        checkpoint_sha=str(z["checkpoint_sha"]),
        fingerprint=stored,
    )


def main() -> None:
    import argparse

    from birdcam.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="student_best.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config()
    written = run(cfg, checkpoint_name=args.checkpoint, batch_size=args.batch_size)
    for kind, path in written.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
