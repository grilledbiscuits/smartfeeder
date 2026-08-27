"""Serialise the open-set failsafe so the capture app can actually construct it.

Why this exists
---------------
`capture/classifier.py:build_novelty_scorer` accepts `scorer: knn` and then
refuses, because the kNN scorer needs reference feature vectors on disk and
nothing ever wrote them (A26, and the external audit's finding 3). The result
is that deployment falls back to the `energy` scorer -- AUROC 0.927 against
kNN's 0.979 on web OOD -- purely for want of a file. This module writes that
file.

Two decisions here are not obvious.

**Features come from the FROZEN backbone, not the fine-tuned checkpoint.**
A25 measured that fine-tuning collapses the feature geometry a distance test
depends on: kNN AUROC 0.979 -> 0.903, catch rate at 5% FPR 0.909 -> 0.431.
Measured again on real footage 2026-08-27, the collapse is worse than the AUROC
suggests -- fine-tuned novelty scores across 7,588 uncut frames spanned 0.565
(median) to 0.649 (max) against a 0.546 threshold, a dynamic range of roughly
0.08. A detector that returns nearly the same number for an empty feeder and a
sunbird is not detecting anything. The same frames scored through the frozen
backbone span 0.420-0.799, and rank empty feeder (median 0.723) above real
target birds (0.560) -- actual discrimination.

**The threshold is calibrated on FIELD birds, not web data.** Calibrating a
15% false-alarm rate on web validation puts the threshold at 0.507, which
rejects 71% of real double-collared frames -- the failsafe would suppress most
genuine visits. A27 established that this system misses birds rather than
recording squirrels, so the operating point is chosen the other way round: pick
the threshold that rejects at most `target_reject` of real field bird frames,
and report what that costs in empty-feeder suppression rather than assuming it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PHASE = 8

# Field folders holding real target-bird footage. `uncut` is excluded: it is
# ~58% empty feeder and is the closest thing to a negative set we have.
BIRD_FOLDERS = ("doublecollared", "capewhiteeye", "juvenileamethyst", "amethyst", "capebulbul")


def frozen_features(cfg, paths: list[str], batch_size: int = 32) -> np.ndarray:
    """Pooled features from the ImageNet-pretrained backbone, no fine-tuning."""
    import timm
    import torch
    from PIL import Image

    from birdcam.train.augment import build_eval_transform

    name = cfg.train_cfg["backbone"]["student"]["name"]
    size = cfg.train_cfg["backbone"]["student"]["image_size"]
    model = timm.create_model(name, pretrained=True, num_classes=0).eval()
    tf = build_eval_transform(size)

    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch = torch.stack(
                [tf(Image.open(cfg.root / p).convert("RGB")) for p in paths[i : i + batch_size]]
            )
            out.append(model(batch).float().numpy())
            if i and i % (batch_size * 40) == 0:
                logger.info("  %d/%d frames", i, len(paths))
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def calibrate_on_field(scorer, bird_scores: np.ndarray, target_reject: float) -> float:
    """Threshold rejecting at most `target_reject` of real field bird frames.

    Deliberately the inverse of `NoveltyScorer.calibrate`, which fixes a false
    ALARM rate against in-distribution web data. In deployment the expensive
    error is suppressing a genuine visit, so the rate that gets pinned is the
    one measured on real birds.
    """
    return float(np.quantile(bird_scores, 1.0 - target_reject))


def build(cfg, target_reject: float = 0.10, max_reference: int = 1000, k: int = 10) -> Path:
    """Fit the scorer on frozen web features, calibrate on field birds, write it."""
    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest
    from birdcam.models.novelty import KNNScorer

    sweep = cfg.path("embeddings_dir") / "sweep"
    name = cfg.train_cfg["backbone"]["student"]["name"]
    # timm tags the cache with its pretrained suffix (`tf_efficientnetv2_b0.in1k_18146.npy`)
    # while the config carries the bare architecture name, so match on the prefix.
    stem = name.replace("/", "_")
    cands = sorted(p for p in sweep.glob(f"{stem}*.npy") if not p.name.startswith("ood_"))
    if not cands:
        raise FileNotFoundError(
            f"no frozen embedding cache for {name} in {sweep}. Run the backbone "
            "sweep first -- the frozen features are the whole point of this bundle."
        )
    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)

    # Pick the cache whose row count matches the manifest, NOT the last one
    # sorted: the sweep leaves several generations behind, and string order puts
    # `..._9318.npy` after `..._18146.npy`, so sorting silently selects a
    # Tier-A-only cache from an earlier phase.
    chosen = next((p for p in cands if len(np.load(p, mmap_mode="r")) == len(items)), None)
    if chosen is None:
        sizes = {p.name: len(np.load(p, mmap_mode="r")) for p in cands}
        raise RuntimeError(
            f"no frozen cache matches the manifest's {len(items)} rows. Found {sizes}. "
            "Re-run the backbone sweep."
        )
    logger.info("frozen features: %s", chosen.name)
    frozen_web = np.load(chosen)
    split = np.array([i.split for i in items])
    y = np.array([i.taxon_index for i in items])

    tr = split == "train"
    scorer = KNNScorer(k=k, max_reference=max_reference).fit(frozen_web[tr], y[tr])

    # Field bird frames, scored through the same frozen backbone.
    pred = cfg.path("data_root") / "field" / "predictions.npz"
    if not pred.is_file():
        raise FileNotFoundError(f"no field predictions at {pred}; run birdcam.data.field first")
    z = np.load(pred, allow_pickle=True)
    folders = z["folders"].astype(str)
    paths = z["paths"].astype(str)
    bird_mask = np.isin(folders, BIRD_FOLDERS)
    logger.info("scoring %d field bird frames through the frozen backbone", int(bird_mask.sum()))
    bird_scores = scorer.score(frozen_features(cfg, list(paths[bird_mask])))

    threshold = calibrate_on_field(scorer, bird_scores, target_reject)

    # What that costs on the closest thing to a negative set we have.
    uncut = paths[folders == "uncut"]
    sample = list(uncut[:: max(1, len(uncut) // 1500)])
    uncut_scores = scorer.score(frozen_features(cfg, sample))
    flagged_uncut = float((uncut_scores > threshold).mean())

    ref = scorer._ref
    out = cfg.path("data_root") / "export" / "novelty_knn.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "backbone": name,
        "features": "frozen ImageNet pretrained, NOT the fine-tuned checkpoint (A25)",
        "k": k,
        "n_reference": int(len(ref)),
        "threshold": threshold,
        "calibration": f"rejects {target_reject:.0%} of field bird frames",
        "measured_field_bird_reject": float((bird_scores > threshold).mean()),
        "measured_uncut_flagged": flagged_uncut,
        "reference_sha": hashlib.sha256(np.ascontiguousarray(ref).tobytes()).hexdigest()[:16],
    }
    np.savez_compressed(out, features=ref, threshold=threshold, meta=json.dumps(meta))
    logger.info("wrote %s", out)
    for key, val in meta.items():
        logger.info("  %-26s %s", key, val)
    return out


def main() -> None:
    import argparse

    from birdcam.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target-reject",
        type=float,
        default=0.10,
        help="fraction of real field bird frames the gate may reject",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-reference", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config()
    path = build(cfg, target_reject=args.target_reject, k=args.k, max_reference=args.max_reference)
    print(f"\nwrote {path}")
    print("point capture config at it:  classifier.novelty.reference: " + str(path))


if __name__ == "__main__":
    main()
