"""The geometric empty-feeder gate.

It exists because the classifier cannot answer this question: measured
2026-09-11, the `empty_feeder` class scored recall 1.000 on backgrounds it had
trained on and 0.000 on one it had not, having memorised backgrounds rather
than learned what an unoccupied feeder looks like.

The gate's cardinal rule is that it FAILS OPEN. A false "empty" silences a real
visit; a false "occupied" costs one classifier call. Most of these tests are
about that asymmetry.
"""

from __future__ import annotations

import numpy as np
import pytest

from capture.emptygate import DEFAULT_THRESHOLD, EmptyGate, find_port, port_box


def scene(w=320, h=180, port=(200, 90), bird=False) -> np.ndarray:
    """A grey scene with a red feeding port, optionally with a bird at it.

    The bird is sized to cover a realistic share of the port box. On the real
    footage an occupied port departs from its background by about 64 (mean
    absolute difference, 0-255) against 2.9 when empty; a bird drawn as a few
    hundred pixels inside a 100x100 box would average out to 3 and make these
    tests pass or fail on the drawing, not on the gate.
    """
    a = np.full((h, w, 3), 120.0, dtype=np.float32)
    px, py = port
    a[py - 6 : py + 6, px - 9 : px + 9] = (205, 30, 38)          # the port
    if bird:
        a[py - 34 : py + 26, px - 46 : px + 14] = (35, 70, 45)   # a bird at the port
    return a


def fill(gate: EmptyGate, n: int = 40, **kw) -> None:
    for _ in range(n):
        gate.observe(scene(**kw))


# --- finding the port ---------------------------------------------------------


def test_finds_the_red_port() -> None:
    cx, cy = find_port(scene())
    assert abs(cx - 200) <= 5 and abs(cy - 90) <= 5


def test_no_port_returns_none_rather_than_guessing() -> None:
    """A wrong port location silently moves the search window off the bird."""
    assert find_port(np.full((180, 320, 3), 120.0, dtype=np.float32)) is None
    assert port_box(np.full((180, 320, 3), 120.0, dtype=np.float32)) is None


def test_port_box_stays_in_frame_wherever_the_port_is() -> None:
    """Shifted, never clipped -- clipping put the port at its own box edge."""
    for px in (20, 160, 300):
        box = port_box(scene(port=(px, 90)))
        assert box is not None
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 180
        assert x0 <= px < x1, "the port fell outside its own box"


# --- failing open -------------------------------------------------------------


def test_not_empty_before_a_background_exists() -> None:
    """An unwarmed gate must pass every frame through."""
    g = EmptyGate()
    assert not g.ready
    assert not g.is_empty(scene())


def test_not_empty_when_the_port_cannot_be_found() -> None:
    g = EmptyGate()
    for _ in range(40):
        g.observe(np.full((180, 320, 3), 120.0, dtype=np.float32))
    assert g.ready
    assert g.score(np.full((180, 320, 3), 120.0, dtype=np.float32)) is None
    assert not g.is_empty(np.full((180, 320, 3), 120.0, dtype=np.float32))


def test_not_empty_on_a_frame_of_the_wrong_shape() -> None:
    g = EmptyGate()
    fill(g)
    assert not g.is_empty(scene(w=640, h=360))


def test_empty_clip_requires_every_frame_empty() -> None:
    """A bird arriving in the last second of a clip is still a visit."""
    g = EmptyGate()
    fill(g)
    assert g.clip_is_empty([scene(), scene(), scene()])
    assert not g.clip_is_empty([scene(), scene(), scene(bird=True)])


def test_no_frames_is_not_empty() -> None:
    g = EmptyGate()
    fill(g)
    assert not g.clip_is_empty([])


# --- the actual discrimination ------------------------------------------------


def test_separates_empty_from_occupied() -> None:
    g = EmptyGate()
    fill(g)
    assert g.is_empty(scene())
    assert not g.is_empty(scene(bird=True))
    assert g.score(scene(bird=True)) > DEFAULT_THRESHOLD > g.score(scene())


def test_background_survives_a_bird_in_a_minority_of_frames() -> None:
    """The median is what lets the background be built from live footage."""
    g = EmptyGate()
    for i in range(40):
        g.observe(scene(bird=(i % 4 == 0)))   # occupied a quarter of the time
    assert g.is_empty(scene())
    assert not g.is_empty(scene(bird=True))


def test_moving_the_camera_resets_the_background() -> None:
    """An old background describes a scene that no longer exists."""
    g = EmptyGate()
    fill(g)
    assert g.ready
    g.observe(scene(w=640, h=360))
    assert not g.ready, "buffer should have been cleared by the shape change"


def test_observing_is_safe_on_occupied_frames() -> None:
    """Callers feed every frame; requiring known-empty input defeats the point."""
    g = EmptyGate()
    for _ in range(40):
        g.observe(scene(bird=True))
    # Background is now the occupied scene, so an occupied frame reads as empty.
    # That is expected and is why the buffer spans a long, sparsely sampled
    # window -- the guarantee is only that observe() never raises.
    assert g.ready


@pytest.mark.parametrize("thr", [0.0, 1e9])
def test_threshold_is_the_only_knob(thr) -> None:
    g = EmptyGate(threshold=thr)
    fill(g)
    assert g.is_empty(scene()) == (thr > 0)
