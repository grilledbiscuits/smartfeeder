"""Is anything at the feeder at all? Decided geometrically, not by the model.

The classifier cannot answer this and the reason is structural. Measured
2026-09-11 on the trained checkpoint: the `empty_feeder` class scored recall
**1.000** on the nine session backgrounds it trained on and **0.000** on a
background it had not seen, where it called every empty frame a Cape White-eye.
It had memorised backgrounds rather than learning what an unoccupied feeder
looks like -- an unsurprising shortcut, since an empty-feeder frame is nothing
*but* background. It was also actively harmful: all seven of its validation
predictions were false positives on real subjects, each of which it would have
declined to record.

The fix is to stop asking a classifier a question the geometry answers better.
**The camera is fixed.** That fact is thrown away by treating every frame as an
independent image, and it is exactly what makes background subtraction work:

    empty frames   median departure   2.93
    occupied       median departure  64.02

    one threshold at 12.01 -> 98.9% of empty frames, 97.5% of occupied ones,
    measured across all nine recording sessions.

Two details carry that separation, and both were learned the hard way while
mining training negatives:

* **Look only at the feeding port.** A sunbird is a percent or two of the frame
  and the shade cloth behind the feeder moves in wind across all of it, so a
  whole-frame score loses the bird in the noise. Restricted to the port, the
  bird *is* the signal. The port needs no model to find: it is the only strongly
  red thing in the scene.

* **Take a MEDIAN over many frames for the background, never a mean.** The
  median survives birds being present in a minority of frames, which is what
  lets the background be maintained from live footage rather than needing a
  known-empty reference.

## Failing open

Every uncertain path here returns "not empty". A false *empty* silences a real
visit, which is the one error this system must not make; a false *occupied*
costs one classifier call. So a missing background, an undetectable port or an
unreadable frame all pass the frame through.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Departure at the port, as mean absolute difference over 0-255 RGB. Derived as
# the 99th percentile of 476 individually verified empty frames; the 5th
# percentile of occupied frames sits at 13.4, so the gap is real but narrow at
# the tails. Raise it to suppress more, lower it to be safer -- but see "Failing
# open": the asymmetry is deliberate.
DEFAULT_THRESHOLD = 12.0

# Frames held for the rolling background. Sampled sparsely over a long window so
# a bird that perches for several minutes is still a minority of the buffer.
DEFAULT_BACKGROUND_FRAMES = 48

# Fraction of the longer image edge used as the half-width of the port box.
PORT_BOX_FRAC = 0.16


def find_port(background: np.ndarray) -> tuple[int, int] | None:
    """Centre of the red feeding port, or None if it cannot be located.

    Red minus the larger of green and blue, thresholded at the far tail.
    Everything else in this scene -- shade cloth, brick, foliage, sky -- is
    green, grey or blue, so the port is separable without a model. A male
    sunbird's red breast band cannot move this: it is one frame's worth of
    pixels against a median, and far smaller than the port.
    """
    if background.ndim != 3 or background.shape[2] < 3:
        return None
    r, g, b = background[..., 0], background[..., 1], background[..., 2]
    redness = r - np.maximum(g, b)
    mask = redness > max(30.0, float(np.percentile(redness, 99.5)))
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    return int(xs.mean()), int(ys.mean())


def _window(centre: int, half: int, limit: int) -> tuple[int, int]:
    """A fixed-width window kept in frame by SHIFTING it, never clipping.

    Clipping is what made an earlier version of this miss birds: with the port
    near a frame edge, a clipped box puts the port at its own edge and a bird on
    the far side falls outside the box entirely.
    """
    half = min(half, limit // 2)
    lo = max(0, min(centre - half, limit - 2 * half))
    return lo, lo + 2 * half


def port_box(
    background: np.ndarray, frac: float = PORT_BOX_FRAC
) -> tuple[int, int, int, int] | None:
    """A tight box around the port, guaranteed to lie inside the frame."""
    found = find_port(background)
    if found is None:
        return None
    cx, cy = found
    h, w = background.shape[:2]
    half = int(frac * max(w, h))
    if half < 4:
        return None
    x0, x1 = _window(cx, half, w)
    y0, y1 = _window(cy, half, h)
    return x0, y0, x1, y1


class EmptyGate:
    """Rolling background for one fixed camera, and the check against it.

    Feed it frames with `observe`; ask it with `is_empty`. It is safe to observe
    every frame, occupied or not -- the median is what makes that work, and
    feeding only known-empty frames would defeat the point of maintaining a
    background from live footage.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        background_frames: int = DEFAULT_BACKGROUND_FRAMES,
    ) -> None:
        self.threshold = float(threshold)
        self._buf: deque[np.ndarray] = deque(maxlen=int(background_frames))
        self._background: np.ndarray | None = None
        self._box: tuple[int, int, int, int] | None = None

    @property
    def ready(self) -> bool:
        """Whether enough frames have been seen to trust the background."""
        return len(self._buf) >= max(8, self._buf.maxlen // 4)

    def observe(self, frame: np.ndarray) -> None:
        """Add a frame to the rolling background.

        A frame whose shape differs from the buffer resets it: the camera has
        been changed or reoriented, and the old background describes a scene
        that no longer exists.
        """
        a = np.asarray(frame, dtype=np.float32)
        if a.ndim != 3:
            return
        if self._buf and a.shape != self._buf[0].shape:
            logger.info(
                "frame shape changed %s -> %s; resetting background",
                self._buf[0].shape,
                a.shape,
            )
            self._buf.clear()
        self._buf.append(a)
        self._background = None  # recomputed lazily
        self._box = None

    def background(self) -> np.ndarray | None:
        if not self.ready:
            return None
        if self._background is None:
            self._background = np.median(np.stack(self._buf), axis=0)
            self._box = port_box(self._background)
        return self._background

    def score(self, frame: np.ndarray) -> float | None:
        """Mean absolute departure at the port, or None if it cannot be measured."""
        bg = self.background()
        if bg is None or self._box is None:
            return None
        a = np.asarray(frame, dtype=np.float32)
        if a.shape != bg.shape:
            return None
        x0, y0, x1, y1 = self._box
        return float(np.abs(a[y0:y1, x0:x1] - bg[y0:y1, x0:x1]).mean())

    def is_empty(self, frame: np.ndarray) -> bool:
        """True only when the port is measurably unoccupied. Fails open."""
        s = self.score(frame)
        if s is None:
            return False
        return s < self.threshold

    def clip_is_empty(self, frames: list[np.ndarray]) -> bool:
        """True only if EVERY frame of the clip is empty.

        A bird arriving in the last second of a clip is a visit. One occupied
        frame is enough to send the whole clip to the classifier.
        """
        if not frames:
            return False
        return all(self.is_empty(f) for f in frames)


def load_frame(path: Path, max_side: int = 320) -> np.ndarray | None:
    """Read one frame as RGB, downscaled just enough to keep the port legible.

    The port box is about a third of the longer edge, so at 320px it is still
    ~100px across -- ample for a mean-absolute-difference, and a quarter of the
    arithmetic of full resolution.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
            return np.asarray(im, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - a bad frame must not stop a capture
        logger.warning("could not read %s for the empty gate: %s", path, exc)
        return None
