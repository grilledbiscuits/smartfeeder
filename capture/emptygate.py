"""Is anything at the feeding port? Decided geometrically, not by the model.

The classifier cannot answer this: its `empty_feeder` class memorised session
backgrounds (recall 1.000 on seen backgrounds, 0.000 on an unseen one), and with
that class suppressed the Pi replay published all four empty clips as birds.

The camera is fixed, so compare the port region against a background. The hard
part is where the background comes from:

* **Not from triggered clips.** A median over PIR-triggered frames assumes the
  foreground is a minority; at a busy feeder it is not, and a regular visitor
  becomes the background and is discarded as "empty" (reproduced 2026-09-16).
* **Not from one fixed reference.** Outdoors the light moves all day; a single
  morning frame stops matching by afternoon and the gate silently never fires.

`IdleBackgroundGate` takes a snapshot every few minutes while the PIR has been
quiet, and medians the last few. Triggered frames never touch it. It tracks the
light, and a bird perched during one snapshot is outvoted by the others.

## Failing open

Every uncertain path returns "not empty": no fresh background (e.g. a feeder
busy for longer than `max_age_seconds`), no port found, a frame shape that does
not match the snapshots, an unreadable frame. A false empty silences a real
visit; a false occupied costs one classifier call.

Only the port region is compared, so a subject away from the port can be missed.
The threshold (12.0, measured on static session backgrounds) must be revalidated
with installed-camera snapshots, which come from the camera's preview stream
rather than decoded H.264.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
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

# Idle snapshots: one every `interval` once the PIR has been quiet for `quiet`
# seconds; median of the last `frames`, trusted for `max_age` after the newest.
# Six at five minutes spans half an hour of light.
DEFAULT_IDLE_FRAMES = 6
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 300.0
DEFAULT_QUIET_SECONDS = 120.0
DEFAULT_MAX_AGE_SECONDS = 900.0

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

    Experimental only. A median assumes the foreground is absent in most
    observations, which PIR-triggered clips do not guarantee. Use the fixed
    TrustedEmptyGate through the production builder instead.
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
            self._box = None  # the port must be re-found in the new framing
        self._buf.append(a)
        self._background = None  # recomputed lazily, port box reused

    def background(self) -> np.ndarray | None:
        """The port-box background, recomputed only when the buffer has changed.

        Only the port box is ever scored, so only the port box is reduced. The
        first pass has to take the median of whole frames -- the port has to be
        found before it can be cropped to -- but every pass after that medians
        the crop alone, which is about a sixth of the pixels.

        That is not a micro-optimisation. Measured on an x86 laptop, a full-frame
        median over 48 frames at 320px costs 295 ms; a Pi 4B is several times
        slower again, and this runs on every motion event before the recorder is
        started. A gate that costs a second to say "nothing there" would spend
        more than it saves.
        """
        if not self.ready:
            return None
        if self._box is None:
            full = np.median(np.stack(self._buf), axis=0)
            self._box = port_box(full)
            if self._box is None:
                return None
        if self._background is None:
            x0, y0, x1, y1 = self._box
            self._background = np.median(np.stack([f[y0:y1, x0:x1] for f in self._buf]), axis=0)
        return self._background

    def score(self, frame: np.ndarray) -> float | None:
        """Mean absolute departure at the port, or None if it cannot be measured."""
        bg = self.background()
        if bg is None or self._box is None:
            return None
        a = np.asarray(frame, dtype=np.float32)
        if a.shape[:2] != self._buf[0].shape[:2]:
            if not getattr(self, "_warned_shape", False):
                # Otherwise the gate is silently off for every clip.
                logger.warning(
                    "empty gate disabled for frames of shape %s: background is %s",
                    a.shape[:2],
                    self._buf[0].shape[:2],
                )
                self._warned_shape = True
            return None
        x0, y0, x1, y1 = self._box
        return float(np.abs(a[y0:y1, x0:x1] - bg).mean())

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


class IdleBackgroundGate(EmptyGate):
    """Background from quiet-period snapshots only; triggered clips never update it."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        background_frames: int = DEFAULT_IDLE_FRAMES,
        *,
        snapshot_interval_seconds: float = DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
        quiet_seconds: float = DEFAULT_QUIET_SECONDS,
        max_age_seconds: float | None = DEFAULT_MAX_AGE_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("empty threshold must be positive and finite")
        if int(background_frames) < 1:
            raise ValueError("background_frames must be at least 1")
        super().__init__(threshold=threshold, background_frames=background_frames)
        self.snapshot_interval = float(snapshot_interval_seconds)
        self.quiet = float(quiet_seconds)
        self.max_age = None if max_age_seconds is None else float(max_age_seconds)
        self._monotonic = monotonic
        self._last_snapshot = float("-inf")
        self._newest = float("-inf")

    @property
    def min_frames(self) -> int:
        return min(3, self._buf.maxlen)

    @property
    def ready(self) -> bool:
        if len(self._buf) < self.min_frames:
            return False
        return self.max_age is None or self._monotonic() - self._newest <= self.max_age

    def observe(self, frame: np.ndarray) -> None:
        """Triggered frames are ignored: a regular visitor must never become background."""

    def snapshot_due(self, last_motion: float) -> bool:
        now = self._monotonic()
        return (
            now - self._last_snapshot >= self.snapshot_interval and now - last_motion >= self.quiet
        )

    def observe_idle(self, frame: np.ndarray | None) -> None:
        """Add a quiet-period snapshot. `None` (a failed grab) still resets the timer."""
        self._last_snapshot = self._monotonic()
        if frame is None:
            return
        a = shrink(frame)
        if a.ndim != 3 or a.shape[2] != 3 or not np.isfinite(a).all():
            logger.warning("empty gate ignored a malformed snapshot of shape %s", a.shape)
            return
        super().observe(a)
        self._newest = self._last_snapshot
        self._warned_shape = False

    def seed(self, frame: np.ndarray) -> None:
        """Fill the buffer from one known-empty frame (replay and testing)."""
        for _ in range(self._buf.maxlen):
            self.observe_idle(frame)
        if self.background() is None:
            raise ValueError("feeding port not found in the seed frame")


def shrink(frame: np.ndarray, max_side: int = 320) -> np.ndarray:
    """Downscale an RGB array exactly as `load_frame` downscales a sampled JPEG."""
    a = np.asarray(frame)
    if a.ndim != 3:
        return a
    if max(a.shape[:2]) <= max_side:
        return np.asarray(a, dtype=np.float32)
    from PIL import Image

    im = Image.fromarray(np.clip(a[..., :3], 0, 255).astype(np.uint8))
    scale = max_side / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
    return np.asarray(im, dtype=np.float32)


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
