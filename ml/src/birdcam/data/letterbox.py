"""Find field frames that are portrait video padded into a landscape frame.

Four recording sessions were filmed with the phone held upright and saved as
455x256 frames with black pillarbox bars: 68% of every frame is black and the
bird occupies a 145px-wide strip. They were extracted, labelled and trained on
without anyone checking the geometry.

Why it matters
--------------
The deployment camera never produces bars, so these frames are a distribution
the Pi will never see, and they are not neutral:

* Measured 2026-09-21 on `student_best.pt`, Southern Double-collared Sunbird
  scored 0.998 val / 0.989 test on unbarred frames and 0.828 / 0.877 on barred
  ones -- 47 of 400 barred test frames were called `other_animal`, which is
  exactly the failure the Pi replay soak surfaced.
* They were 43% of validation and 47% of test, so every field metric quoted
  before this date averages over a distribution that does not exist at the
  feeder, and understates the real one.
* Barred frames look stranger to the model (median energy -5.56 against -6.62),
  which pushed the fitted novelty threshold out to -4.76 where unbarred frames
  alone give -5.01.

Detected per FRAME, not per session and not from a hard-coded list. Session
20260829_072441 turned out to hold both kinds -- 68%-barred landscape frames and
genuinely portrait ones -- so a per-session rule would have thrown away good
frames or kept bad ones. New footage can arrive either way.

The bars are detected on the FRAME, so a frame that happens to be dark all over
would look barred. That is why the test is columns-only, requires the dark
columns to sit at both edges and the centre to be bright: a genuinely dark night
frame fails the centre test and is kept.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# A bar column is near-black; the content must be clearly brighter than that.
DARK_LEVEL = 10.0
BRIGHT_LEVEL = 30.0
# Below this share of dark edge columns it is a dark scene, not a pillarbox.
MIN_BAR_FRACTION = 0.15
# The picture between the bars must be a plausible share of the frame. Without
# this, one bright patch in an otherwise dark night frame reads as "content
# between bars" and the frame is wrongly excluded.
MIN_CONTENT_FRACTION = 0.10


def bar_columns(frame: np.ndarray) -> tuple[int, int] | None:
    """Columns `(left, right)` to crop, or None if the frame is not pillarboxed.

    Returns the bounds of the bright content, so `frame[:, left:right]` is the
    original portrait picture.
    """
    a = np.asarray(frame, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2 or a.shape[1] < 16:
        return None

    column = a.mean(axis=0)
    dark = column < DARK_LEVEL
    if not (dark[0] and dark[-1]):
        return None  # bars sit at both edges or it is not a pillarbox
    if dark.mean() < MIN_BAR_FRACTION or dark.all():
        return None

    left = int(np.argmax(~dark))
    right = int(len(dark) - np.argmax(~dark[::-1]))
    if right - left < 16 or (right - left) < MIN_CONTENT_FRACTION * len(dark):
        return None
    if column[left:right].mean() < BRIGHT_LEVEL:
        return None  # a dark frame all through, not content between bars
    if dark[left:right].mean() > 0.5:
        return None  # "bars" with as much dark inside: not a clean pillarbox
    return left, right


def is_letterboxed(frame: np.ndarray) -> bool:
    return bar_columns(frame) is not None


def _read(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - a bad frame must not stop a run
        logger.warning("could not read %s: %s", path, exc)
        return None


def session_of(item) -> str:
    """The recording session an item came from: `20260828_155630`."""
    return "_".join(item.observation_id.split("_")[:2])


def letterboxed_ids(items, cache_path: Path | None = None) -> set[str]:
    """Image ids of pillarboxed field frames, read once and cached.

    Every field frame is examined: a session is not a safe unit (see above).
    Extracted frames never change, so the cache is keyed by image id alone.
    """
    import json

    cache: dict[str, bool] = {}
    if cache_path and cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception as exc:  # noqa: BLE001 - a corrupt cache is rebuilt
            logger.warning("ignoring unreadable letterbox cache %s: %s", cache_path, exc)

    fresh = 0
    for item in items:
        if getattr(item, "source", None) != "field" or item.image_id in cache:
            continue
        frame = _read(item.path)
        cache[item.image_id] = frame is not None and is_letterboxed(frame)
        fresh += 1

    if cache_path and fresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, sort_keys=True))
    return {image_id for image_id, barred in cache.items() if barred}


def is_deployment_framing(items, cache_path: Path | None = None):
    """A predicate over items: True for frames framed like the Pi's camera.

    Web images are always True -- they are a different domain on purpose, and
    the point here is only to exclude field frames whose geometry is an artefact
    of how the phone was held.
    """
    barred = letterboxed_ids(items, cache_path=cache_path)
    if barred:
        field = sum(1 for i in items if getattr(i, "source", None) == "field")
        logger.info("excluding %d of %d field frames as pillarboxed", len(barred), field)

    def predicate(item) -> bool:
        return getattr(item, "source", None) != "field" or item.image_id not in barred

    return predicate
