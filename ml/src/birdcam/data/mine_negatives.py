"""Mine empty-feeder negatives from uncut footage, without using the classifier.

Why not just ask the model
--------------------------
The negative classes (`empty_feeder`, `insect`, `other_animal`, `obstruction`)
have never had a single positive training example. A27 measured the
consequence: across 126 minutes of mostly-empty feeder the model called a
negative class 0.0% of the time, and the novelty gate absorbed empty frames
instead -- a distance test standing in for a class that could simply be
trained. Using that same model to find the examples would be circular, and
after the Fork-tailed Drongo was added its head is the wrong width anyway.

The method
----------
The feeder is bolted in place and the camera does not move, so within one
recording session almost every pixel is background almost all of the time. A
per-session MEDIAN over sampled frames therefore reconstructs the empty scene
even when no single frame is empty -- a bird occupies one part of the frame for
a minority of the time, and the median discards it.

Each frame is then scored by how far it departs from that background. Empty
frames score near zero; a bird at the port scores high. This is independent of
the classifier, needs no labels, and degrades gracefully: if a session's camera
did move, the background is poor and EVERY frame scores high, which shows up as
an implausible empty-rate rather than as silent mislabelling.

What it deliberately does not do
--------------------------------
It does not label anything as a species, and it does not decide the threshold
for you. It reports the score distribution and a proposed split; the frames
either side must be eyeballed before they become training data. The last time
candidates were taken on trust in this project -- automated white-eye visit
cutting -- verification found 72% precision and two of eighteen samples were
the wrong bird entirely.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PHASE = 8

BACKGROUND_SAMPLE = 240  # frames per session used to build the median


def _load_small(path: Path, size: int = 128) -> np.ndarray:
    """Small RGB array. Detail is irrelevant: we want presence, not identity.

    RGB, not greyscale. Greyscale was the first attempt and it loses exactly the
    birds that matter here: a green-yellow Cape White-eye against green shade
    cloth, or a drab brown sunbird against mid-toned tarp, differ in HUE far
    more than in luminance. Measured on 24 frames the greyscale scorer called
    empty, a third held a bird.
    """
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32)


def session_background(frames: list[Path], sample: int = BACKGROUND_SAMPLE) -> np.ndarray:
    """Median frame for one session -- the empty scene, reconstructed.

    Works even when no individual frame is empty, provided no single position is
    occupied in most frames. At a feeder that holds comfortably.
    """
    step = max(1, len(frames) // sample)
    stack = np.stack([_load_small(p) for p in frames[::step][:sample]])
    return np.median(stack, axis=0)


def _block_max(diff: np.ndarray, block: int = 16) -> float:
    """Largest block-mean departure, not the whole-frame mean.

    A whole-frame mean was the first attempt and it fails in the one way that
    matters: a bird at the port occupies a small share of the frame, so its
    signal averages away against unchanged background. Verified on 24 frames
    the mean called empty -- four held a bird, 83% precision, which as training
    data would teach the model that a sunbird at the port IS an empty feeder.

    Pooling into blocks and taking the maximum makes the score local. One
    occupied block is enough, however quiet the rest of the frame.
    """
    d = diff.mean(axis=2) if diff.ndim == 3 else diff
    h, w = d.shape
    h2, w2 = h // block * block, w // block * block
    tiles = d[:h2, :w2].reshape(h2 // block, block, w2 // block, block)
    return float(tiles.mean(axis=(1, 3)).max())


def score_frames(frames: list[Path], background: np.ndarray) -> np.ndarray:
    """Localised departure from the session background, per frame."""
    return np.array([_block_max(np.abs(_load_small(p) - background)) for p in frames])


def propose_threshold(scores: np.ndarray, empty_percentile: float = 0.15) -> float:
    """Threshold at a conservative per-session percentile of the score.

    Otsu's method was the first attempt and it fails here. It assumes the score
    histogram is bimodal; in practice a session's scores are heavily skewed with
    a long thin tail, so the inter-class variance split lands far out in that
    tail and calls almost everything empty -- one 2,022-frame session came back
    94.8% empty through three different scoring functions.

    Measured precision of the frames below the cut, by eye on 24 random samples:

    On the 2026-08 footage (one fixed camera, 7,588 frames):

        whole-frame mean + Otsu       83%   (4 of 24 held a bird)
        block-max + Otsu              67%   (8 of 24)
        block-max, RGB, bottom 30%    96%   (1 of 24)

    On the 2026-09 footage (13,475 frames), the same scorer does WORSE:

        bottom 30%                    83%   (4 of 24)
        bottom 15%                    92%   (2 of 24)

    The newer sessions are harder: the camera was repositioned between them, the
    crop is tighter, and backgrounds vary from shade cloth to brick to foliage.
    Both residual failures at 15% are low-contrast -- a dark Amethyst Sunbird
    against shaded tarp, a Cape White-eye against green -- which are precisely
    the frames a classifier finds hardest, so mislabelling them as empty is
    worse than the raw rate suggests.

    92% is NOT clean enough to treat as gold labels. Use these as candidates for
    human review, not as finished training data; a person can flick through
    2,000 thumbnails far faster than this can be tuned, and each further tuning
    pass risks fitting the threshold to the samples already inspected.

    Recall is deliberately sacrificed. These frames become `empty_feeder`
    training examples, and a negative class poisoned with birds is worse than a
    smaller clean one -- it would teach the model that a sunbird at the port is
    an empty feeder. There is no shortage of candidate frames.
    """
    return float(np.quantile(scores, empty_percentile))


def run(cfg, frames_root: Path | None = None, empty_percentile: float = 0.15) -> dict:
    """Score every extracted uncut frame and propose an empty/occupied split."""
    root = frames_root or (cfg.path("data_root") / "field" / "frames" / "uncut")
    # A path given on the command line is relative to the shell's cwd, not the
    # project root, and every frame key below is stored relative to cfg.root.
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no extracted uncut frames at {root}")

    by_session: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.glob("*.jpg")):
        by_session[p.name[:15]].append(p)  # yyyymmdd_hhmmss

    out: dict = {"sessions": {}, "frames": {}}
    all_scores = []
    for session, frames in sorted(by_session.items()):
        bg = session_background(frames)
        s = score_frames(frames, bg)
        thr = propose_threshold(s, empty_percentile)
        empty = s < thr
        out["sessions"][session] = {
            "n_frames": len(frames),
            "threshold": round(thr, 3),
            "empty_fraction": round(float(empty.mean()), 3),
            "score_median": round(float(np.median(s)), 3),
            "score_p05": round(float(np.percentile(s, 5)), 3),
            "score_p95": round(float(np.percentile(s, 95)), 3),
        }
        for p, sc, e in zip(frames, s, empty, strict=True):
            key = p.relative_to(cfg.root) if p.is_relative_to(cfg.root) else p
            out["frames"][str(key)] = {
                "score": round(float(sc), 3),
                "empty": bool(e),
            }
        all_scores.append(s)
        logger.info(
            "%s: %d frames, threshold %.2f, %.1f%% look empty",
            session,
            len(frames),
            thr,
            100 * empty.mean(),
        )

    n_empty = sum(1 for v in out["frames"].values() if v["empty"])
    out["total_frames"] = len(out["frames"])
    out["total_empty"] = n_empty
    dest = cfg.path("data_root") / "field" / "empty_candidates.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info("wrote %s: %d of %d frames look empty", dest, n_empty, out["total_frames"])
    return out


def main() -> None:
    import argparse

    from birdcam.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-root", default=None)
    ap.add_argument(
        "--empty-percentile",
        type=float,
        default=0.30,
        help="per-session fraction taken as empty; precision falls as this rises",
    )
    args = ap.parse_args()
    cfg = load_config()
    res = run(
        cfg,
        Path(args.frames_root) if args.frames_root else None,
        empty_percentile=args.empty_percentile,
    )
    print(f"\n{res['total_empty']} of {res['total_frames']} uncut frames look empty")
    print("VERIFY before using these as labels -- see the module docstring.")


if __name__ == "__main__":
    main()
