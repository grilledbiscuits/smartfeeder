"""Preprocessing: integrity, resize, EXIF, perceptual dedup, grouped splits.

Order matters. Verify, then normalise, then dedup on the normalised image, then
split on groups that cannot leak.

Resizing to 256px short side is a hard requirement rather than an optimisation.
Source images run to 4000px; on a CPU-only machine JPEG decode of a 4000px image
costs more than the forward pass through the backbone, so training would be
bottlenecked on decode rather than compute.

## Why the split grouping is what it is

The naive choice -- random split -- is fiction. Photographers upload bursts: the
same individual bird, same perch, same light, seconds apart. Split those
randomly and the test set is a near-copy of the training set.

Grouping by `observation_id` fixes the burst problem. It does not fix
photographer style: one observer's camera, garden and processing habits recur
across all their observations, and a model can learn the photographer instead of
the bird.

Grouping by `observer_id` alone would fix style but breaks stratification -- an
observer photographs many species, so observer-level groups span classes and a
rare species can be pushed entirely into one split.

So the group key is **(observer_id, scientific_name)**. Every group belongs to
exactly one species, which keeps per-species stratification possible, and since
an observation has one observer and one species, every observation lies wholly
inside one group. Bursts cannot straddle, per-class photographer style cannot
straddle, and stratification still works.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass

from PIL import Image, ImageOps

from birdcam.config import Config, load_config
from birdcam.data.manifest import Manifest, open_manifest

logger = logging.getLogger(__name__)

# Pillow refuses very large images by default (decompression-bomb guard). Our
# sources are legitimate photographs, but keep a sane ceiling.
Image.MAX_IMAGE_PIXELS = 120_000_000


@dataclass
class PreprocessStats:
    processed: int = 0
    skipped_existing: int = 0
    quarantined: int = 0


def _phash(img: Image.Image, hash_size: int = 8) -> str:
    import imagehash

    return str(imagehash.phash(img, hash_size=hash_size))


def process_images(cfg: Config, m: Manifest, force: bool = False) -> PreprocessStats:
    """Decode-check, orient, resize and hash every downloaded image.

    Corrupt files are quarantined rather than raising: a handful of truncated
    downloads in a corpus of thousands must not kill a run that took hours.
    """
    stats = PreprocessStats()
    short_side = cfg.train_cfg["preprocess"]["short_side_px"]
    quality = cfg.train_cfg["preprocess"]["jpeg_quality"]
    out_root = cfg.path("processed_dir")
    quarantine = cfg.path("quarantine_dir")

    rows = list(m.iter_rows("status='downloaded'"))
    logger.info("preprocessing %d images", len(rows))

    for i, row in enumerate(rows, 1):
        src = cfg.root / row["local_path"]
        slug = row["scientific_name"].lower().replace(" ", "_")
        dest = out_root / slug / f"{row['image_id'].replace(':', '_')}.jpg"

        if dest.is_file() and not force and row["phash"]:
            stats.skipped_existing += 1
            continue

        try:
            with Image.open(src) as probe:
                probe.verify()  # cheap structural check; invalidates the handle
            with Image.open(src) as img:
                # Apply EXIF orientation, then drop EXIF entirely. Keeping raw
                # orientation would leave images rotated; keeping EXIF leaks
                # camera and GPS metadata into the corpus.
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                w, h = img.size
                scale = short_side / min(w, h)
                if scale < 1.0:
                    img = img.resize(
                        (max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.Resampling.LANCZOS,
                    )
                ph = _phash(img)
                out_w, out_h = img.size
                dest.parent.mkdir(parents=True, exist_ok=True)
                # save() without exif= writes no EXIF block.
                img.save(dest, "JPEG", quality=quality, optimize=True)

            m.mark(row["image_id"], phash=ph, width=out_w, height=out_h, status_detail=None)
            stats.processed += 1
        except Exception as exc:  # noqa: BLE001 - quarantine, never crash the run
            quarantine.mkdir(parents=True, exist_ok=True)
            try:
                if src.is_file():
                    src.replace(quarantine / src.name)
            except OSError:
                pass
            m.mark(row["image_id"], status="quarantined", status_detail=str(exc)[:200])
            stats.quarantined += 1
            logger.warning("quarantined %s: %s", row["image_id"], exc)

        if i % 250 == 0:
            m.commit()
            logger.info("  %d/%d processed", i, len(rows))

    m.commit()
    return stats


def dedup(cfg: Config, m: Manifest) -> int:
    """Perceptual-hash dedup across the whole corpus, keeping highest resolution.

    Cross-source duplication is guaranteed, not hypothetical: GBIF aggregates
    iNaturalist, so the same photograph arrives twice under different IDs. Even
    within iNaturalist, the same photo is sometimes attached to two observations.

    Bucket by a hash prefix, then sweep within buckets. A full pairwise
    comparison is O(n^2) and unnecessary at this scale.
    """
    threshold = cfg.train_cfg["preprocess"]["phash_hamming_threshold"]
    rows = list(m.iter_rows("phash IS NOT NULL AND status='downloaded'"))
    logger.info("dedup over %d images (hamming <= %d)", len(rows), threshold)

    def to_int(h: str) -> int:
        return int(h, 16)

    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r["phash"][:4]].append(r)

    marked = 0
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        keep: list = []
        # Highest resolution first, so the instance we keep is the best one.
        for r in sorted(bucket, key=lambda x: -((x["width"] or 0) * (x["height"] or 0))):
            h = to_int(r["phash"])
            if any(bin(h ^ to_int(k["phash"])).count("1") <= threshold for k in keep):
                m.mark(r["image_id"], status="duplicate", status_detail="phash duplicate")
                marked += 1
            else:
                keep.append(r)
    m.commit()
    logger.info("marked %d duplicates", marked)
    return marked


def make_splits(cfg: Config, m: Manifest, seed: int | None = None) -> dict[str, int]:
    """Assign train/val/test, grouped and stratified.

    Group key is (observer_id, scientific_name) -- see module docstring. Groups
    are assigned whole, largest first, to whichever split is furthest below its
    target within that stratum. Deterministic given the seed.
    """
    sp = cfg.train_cfg["preprocess"]["split"]
    seed = seed if seed is not None else cfg.train_cfg["compute"]["seed"]
    rng = random.Random(seed)

    # OOD rows are EXCLUDED from the splits entirely. They are an evaluation
    # set for the open-set failsafe, not training data -- a novelty detector
    # fitted on the intruders it is meant to reject would just be a closed-set
    # classifier for known intruders, and would fail on the first novel one.
    rows = list(m.iter_rows("status='downloaded' AND tier != 'OOD'"))
    if not rows:
        raise RuntimeError("no downloaded images to split")
    m.conn.execute("UPDATE images SET split=NULL WHERE tier='OOD'")

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        observer = r["observer_id"] or f"obs:{r['observation_id']}"
        groups[(observer, r["scientific_name"])].append(r)

    # Stratify jointly on species and sex. A group may contain several sex
    # annotations; its stratum takes the rarest present, so scarce female groups
    # drive placement instead of being averaged away by abundant males.
    rarity = {"Female": 0, "Cannot Be Determined": 1, "Male": 2, None: 3}

    def stratum(rs: list) -> tuple[str, str]:
        rarest = min((r["sex_annotation"] for r in rs), key=lambda s: rarity.get(s, 3))
        return (rs[0]["scientific_name"], str(rarest))

    by_stratum: dict[tuple[str, str], list] = defaultdict(list)
    for key, rs in groups.items():
        by_stratum[stratum(rs)].append((key, rs))

    targets = {"train": sp["train"], "val": sp["val"], "test": sp["test"]}
    assigned: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for _strat, entries in sorted(by_stratum.items()):
        rng.shuffle(entries)
        # Largest groups first: greedy placement then keeps the realised
        # proportions closest to target.
        entries.sort(key=lambda e: -len(e[1]))
        counts = {"train": 0, "val": 0, "test": 0}
        total = sum(len(rs) for _, rs in entries)
        for _key, rs in entries:
            deficits = {s: targets[s] * total - counts[s] for s in targets}
            pick = max(deficits, key=lambda s: deficits[s])
            for r in rs:
                m.mark(r["image_id"], split=pick)
            counts[pick] += len(rs)
            assigned[pick] += len(rs)
    m.commit()

    splits_file = cfg.path("splits_file")
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    splits_file.write_text(
        json.dumps(
            {
                "seed": seed,
                "group_key": "(observer_id, scientific_name)",
                "counts": assigned,
                "n_groups": len(groups),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return assigned


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    import argparse

    ap = argparse.ArgumentParser(description="Preprocess, dedup and split the corpus.")
    ap.add_argument("--force", action="store_true", help="reprocess already-processed images")
    ap.add_argument("--skip-dedup", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    with open_manifest(cfg.path("manifest_db")) as m:
        st = process_images(cfg, m, force=args.force)
        print(
            f"processed={st.processed} skipped={st.skipped_existing} quarantined={st.quarantined}"
        )
        if not args.skip_dedup:
            n = dedup(cfg, m)
            print(f"duplicates marked: {n}")
        counts = make_splits(cfg, m)
        total = sum(counts.values())
        print("splits:")
        for s, n in counts.items():
            print(f"  {s:<6} {n:>6}  ({100 * n / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main()
