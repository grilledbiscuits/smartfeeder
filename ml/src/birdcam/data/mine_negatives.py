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


def _load_full(path: Path) -> np.ndarray:
    """Full-resolution RGB. The port box is small; downsampling throws the bird away."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def session_background_full(frames: list[Path], sample: int = BACKGROUND_SAMPLE) -> np.ndarray:
    """Median frame at full resolution, for the port-local gate.

    `session_background` works at 128px, which is right for whole-frame scoring
    and useless here: the port box is a sixth of the frame and a sunbird inside
    it is a handful of pixels once downsampled.
    """
    step = max(1, len(frames) // sample)
    return np.median(np.stack([_load_full(p) for p in frames[::step][:sample]]), axis=0)


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


def _sharpness(a: np.ndarray) -> float:
    """Laplacian variance -- how much fine detail an image holds."""
    g = a.mean(axis=2) if a.ndim == 3 else a
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def background_is_usable(background: np.ndarray, frames: list[Path], min_ratio: float = 0.45):
    """Whether a session's median background can be trusted as the empty scene.

    The whole method assumes a fixed camera. When the camera is moved during a
    recording the median smears across positions and stops resembling any real
    frame, so departures from it mean nothing -- and the failures do not show up
    at the decision boundary where a threshold could catch them. Measured on the
    2026-09 footage, the LOWEST-scoring third of one session's "most definitely
    empty" frames included a large dark bird, plainly visible, because the
    background it was compared against was a blur.

    A smeared median is detectably less sharp than the frames it came from, so
    compare Laplacian variance. Sessions failing this are excluded rather than
    thresholded harder: no cut on a meaningless score is safe.
    """
    ratio = _sharpness(background) / (
        np.median([_sharpness(_load_small(p)) for p in frames[:: max(1, len(frames) // 20)][:20]])
        or 1.0
    )
    return ratio >= min_ratio, float(ratio)


def contrast_is_usable(scores: np.ndarray, min_dynamic_range: float = 2.7):
    """Whether a bird's arrival moves this session's score enough to be seen.

    Background differencing only separates empty from occupied if a bird is the
    largest thing that changes. When the tarp behind the feeder is moving in
    wind, every frame departs from the median by a lot, and a small sunbird at
    the port adds little on top -- so the bottom of the distribution is not the
    empty frames, it is the calm moments, bird or no bird.

    That compresses the score distribution, which is measurable without looking
    at a single image: p95/p05, the ratio of a busy frame to a quiet one.

    Measured 2026-09-10 across ten sessions. Nine scored 3.0 to 9.6 and were
    clean on inspection. One -- 20260825_171725 -- scored 2.40, and roughly a
    quarter of the frames it offered as empty had a bird plainly perched at the
    port. The gap between 2.40 and the next value up (3.00) is where this
    threshold sits.

    HONEST LIMIT: that is ONE confirmed bad session. The statistic separates it
    cleanly and the mechanism is sound, but a single positive example cannot
    establish that 2.7 generalises. It is set to fail safe -- the cost of
    dropping a good session is some empty-feeder frames, of which this project
    now has a surplus; the cost of keeping a bad one is birds labelled as an
    empty feeder, which is the one error the negative class must not contain.
    Revisit when more footage makes a second data point available.
    """
    p05, p95 = np.percentile(scores, [5, 95])
    dyn = float(p95 / p05) if p05 > 0 else float("inf")
    return dyn >= min_dynamic_range, dyn


def feeder_port(background: np.ndarray) -> tuple[int, int]:
    """Centre of the red feeding port, from the session background.

    Birds feed AT the port, and the port is the only strongly red thing in the
    scene, so it can be found without a model: red minus the larger of green and
    blue, thresholded at the far tail. Everything else in this frame -- tarp,
    brick, foliage, sky -- is green, grey or blue.
    """
    r, g, b = background[..., 0], background[..., 1], background[..., 2]
    redness = r - np.maximum(g, b)
    mask = redness > max(30.0, float(np.percentile(redness, 99.5)))
    ys, xs = np.where(mask)
    if len(xs) < 10:
        raise ValueError("no feeder port found in background")
    return int(xs.mean()), int(ys.mean())


def _window(centre: int, half: int, limit: int) -> tuple[int, int]:
    """A window of fixed width that stays in frame by SHIFTING, never clipping.

    Clipping is what made the first version of this miss birds. When the port
    sits near a frame edge, a clipped box puts the port at its own edge, and a
    bird perched on the far side of the port falls outside the box entirely --
    so it contributed nothing to the score and was never seen in review either.
    """
    half = min(half, limit // 2)
    lo = max(0, min(centre - half, limit - 2 * half))
    return lo, lo + 2 * half


def port_box(background: np.ndarray, frac: float = 0.16) -> tuple[int, int, int, int]:
    """A tight box around the port, guaranteed to lie inside the frame."""
    h, w = background.shape[:2]
    cx, cy = feeder_port(background)
    half = int(frac * max(w, h))
    x0, x1 = _window(cx, half, w)
    y0, y1 = _window(cy, half, h)
    return x0, y0, x1, y1


def port_local_z(frames, background, box) -> np.ndarray:
    """Robust z-score of departure from the background WITHIN the port box.

    The whole-frame score cannot see a sunbird: the bird occupies a percent or
    two of the pixels while the tarp behind the feeder moves across all of them,
    so a bird's arrival is lost in the noise. Restricted to the port, the bird
    IS the signal.

    Measured 2026-09-10 on a frame the whole-frame score had passed as empty and
    which holds a Cape White-eye plainly perched at the port: the whole-frame
    score put it at z = -0.69, comfortably inside the "definitely empty" region.
    The port-local score puts it 4th of 1,529. Same frame, same background; the
    difference is entirely where you look.

    Median and MAD rather than mean and standard deviation, because the
    contaminating frames are exactly what would inflate a standard deviation and
    hide themselves.
    """
    x0, y0, x1, y1 = box
    bg = background[y0:y1, x0:x1]
    d = np.array([np.abs(_load_full(p)[y0:y1, x0:x1] - bg).mean() for p in frames])
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) or 1.0
    return (d - med) / (1.4826 * mad)


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


def run(
    cfg,
    frames_root: Path | None = None,
    empty_percentile: float = 0.15,
    drop_top: float = 0.0,
    port_z_max: float = -0.5,
) -> dict:
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
    out["excluded_sessions"] = {}
    for session, frames in sorted(by_session.items()):
        bg = session_background(frames)
        usable, ratio = background_is_usable(bg, frames)
        if not usable:
            out["excluded_sessions"][session] = {
                "n_frames": len(frames),
                "sharpness_ratio": round(ratio, 3),
                "reason": "background smeared -- camera moved during the session",
            }
            logger.warning(
                "%s: EXCLUDED, background sharpness ratio %.2f (camera moved); "
                "%d frames yield no usable negatives",
                session,
                ratio,
                len(frames),
            )
            continue
        s = score_frames(frames, bg)
        ok, dyn = contrast_is_usable(s)
        if not ok:
            out["excluded_sessions"][session] = {
                "n_frames": len(frames),
                "dynamic_range": round(dyn, 3),
                "reason": "score range compressed -- a bird moves it too little to detect",
            }
            logger.warning(
                "%s: EXCLUDED, score dynamic range %.2f (background motion swamps "
                "the bird signal); %d frames yield no usable negatives",
                session,
                dyn,
                len(frames),
            )
            continue
        # Port-local gate. The whole-frame score decides which frames are
        # PLAUSIBLY empty; this decides which of those are certainly empty, by
        # looking only where a bird would be. Run second because it is far more
        # expensive: full-resolution loads, not 128px ones.
        try:
            bg_full = session_background_full(frames)
            box = port_box(bg_full)
        except ValueError:
            logger.warning(
                "%s: EXCLUDED, no feeder port located in the background; "
                "the port-local check cannot run and a whole-frame score alone "
                "has been measured to pass birds",
                session,
            )
            out["excluded_sessions"][session] = {
                "n_frames": len(frames),
                "reason": "feeder port not found; port-local check impossible",
            }
            continue
        thr = propose_threshold(s, empty_percentile)
        empty = s < thr
        # Optional second stage: drop the top slice of what survived.
        # Off by default, because it was measured and it does not do what it was
        # built to do. The hypothesis was that surviving contamination sits at
        # the top of each session's candidate range. Ranked WITHIN session, the
        # riskiest 36 frames were clean; the birds all came from one session
        # whose scores are compressed (see contrast_is_usable), and a global
        # sort by raw score had simply surfaced that session's frames because
        # its scores are numerically larger than everyone else's. Scores are not
        # comparable across sessions and treating them as if they were is what
        # made this look like a boundary problem.
        # Kept as a knob for footage where the picture changes; costing 10% of
        # the negatives for no measured gain is not a default.
        if drop_top > 0 and empty.any():
            kept = np.where(empty)[0]
            cut = propose_threshold(s[kept], 1.0 - drop_top)
            empty = empty & (s < cut)
        out["sessions"][session] = {
            "n_frames": len(frames),
            "threshold": round(thr, 3),
            "empty_fraction": round(float(empty.mean()), 3),
            "drop_top": drop_top,
            "dynamic_range": round(dyn, 3),
            "score_median": round(float(np.median(s)), 3),
            "score_p05": round(float(np.percentile(s, 5)), 3),
            "score_p95": round(float(np.percentile(s, 95)), 3),
        }
        # Anything the whole-frame pass proposed must also be unremarkable at
        # the port. Scored only over the proposals, so the median it is measured
        # against is the median of frames already believed empty.
        idx = np.where(empty)[0]
        if len(idx) >= 8:
            pz = port_local_z([frames[i] for i in idx], bg_full, box)
            empty[idx] = pz <= port_z_max
        out["sessions"][session]["port_box"] = list(box)
        out["sessions"][session]["kept_after_port_gate"] = int(empty.sum())

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
        "--port-z-max",
        type=float,
        default=-0.5,
        help="keep a frame only if its port-local departure is this many robust "
        "standard deviations BELOW its session median; negative is deliberate",
    )
    ap.add_argument(
        "--drop-top",
        type=float,
        default=0.0,
        help="fraction of the SELECTED frames, per session, to trim from the top "
        "of the score range; measured to give no gain, see the note in run()",
    )
    ap.add_argument(
        "--empty-percentile",
        type=float,
        default=0.15,
        help="per-session fraction taken as empty; precision falls as this rises",
    )
    args = ap.parse_args()
    cfg = load_config()
    res = run(
        cfg,
        Path(args.frames_root) if args.frames_root else None,
        empty_percentile=args.empty_percentile,
        drop_top=args.drop_top,
        port_z_max=args.port_z_max,
    )
    print(f"\n{res['total_empty']} of {res['total_frames']} uncut frames look empty")
    print("VERIFY before using these as labels -- see the module docstring.")


if __name__ == "__main__":
    main()
