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
    a[py - 6 : py + 6, px - 9 : px + 9] = (205, 30, 38)  # the port
    if bird:
        a[py - 34 : py + 26, px - 46 : px + 14] = (35, 70, 45)  # a bird at the port
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
        g.observe(scene(bird=(i % 4 == 0)))  # occupied a quarter of the time
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


# --- placement ----------------------------------------------------------------


def _cfg(tmp_path, placement=None):
    """A stand-in with CaptureConfig's strict get(): a missing key raises.

    CaptureConfig.load validates every section, so a real one would fail here
    for reasons that have nothing to do with placement.
    """
    from capture.config import CaptureConfigError

    values = {"empty_gate.enabled": True}
    if placement is not None:
        values["empty_gate.placement"] = placement

    class _Cfg:
        def get(self, dotted):
            if dotted not in values:
                raise CaptureConfigError(f"Missing config key {dotted!r}")
            return values[dotted]

    return _Cfg()


def test_placement_defaults_to_after_record(tmp_path):
    """Measured on a Pi 4B: gating before the recorder cost ~2 s per event."""
    from capture.build import _gate_placement

    assert _gate_placement(_cfg(tmp_path)) == "after_record"


def test_placement_can_be_set_to_before_record(tmp_path):
    """The right trade on a board without a hardware encoder, e.g. a Pi 5."""
    from capture.build import _gate_placement

    assert _gate_placement(_cfg(tmp_path, "before_record")) == "before_record"


def test_unknown_placement_is_refused(tmp_path):
    from capture.build import _gate_placement
    from capture.config import CaptureConfigError

    with pytest.raises(CaptureConfigError):
        _gate_placement(_cfg(tmp_path, "sometimes"))


# --- the production gate: quiet-period snapshots only ---------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def idle_gate(clock=None, **kw):
    from capture.emptygate import IdleBackgroundGate

    return IdleBackgroundGate(monotonic=clock or _Clock(), **kw)


def test_triggered_frames_never_become_background() -> None:
    """Reproduces the 2026-09-16 failure: a regular visitor absorbed as background."""
    g = idle_gate()
    for _ in range(6):
        g.observe_idle(scene())
    bird = scene(bird=True)
    for _ in range(200):
        g.observe(bird)
        assert not g.is_empty(bird)
    assert g.is_empty(scene())


def test_background_tracks_the_light_through_snapshots() -> None:
    """A single morning reference stops matching; rolling snapshots follow."""
    g = idle_gate()
    for _ in range(6):
        g.observe_idle(scene())
    afternoon = scene() * 0.7
    assert not g.is_empty(afternoon), "precondition: the light change exceeds the threshold"
    for _ in range(6):
        g.observe_idle(scene() * 0.7)
    assert g.is_empty(afternoon)
    assert not g.is_empty(afternoon * 0 + scene(bird=True) * 0.7)


def test_a_bird_in_one_snapshot_is_outvoted() -> None:
    g = idle_gate()
    for i in range(6):
        g.observe_idle(scene(bird=(i == 2)))
    assert g.is_empty(scene())
    assert not g.is_empty(scene(bird=True))


def test_stale_background_fails_open() -> None:
    clock = _Clock()
    g = idle_gate(clock, max_age_seconds=900)
    for _ in range(6):
        g.observe_idle(scene())
    assert g.is_empty(scene())
    clock.t += 901
    assert not g.ready
    assert not g.is_empty(scene())
    g.observe_idle(scene())
    assert g.is_empty(scene())


def test_too_few_snapshots_fail_open() -> None:
    g = idle_gate()
    g.observe_idle(scene())
    g.observe_idle(scene())
    assert not g.is_empty(scene())


def test_snapshot_waits_for_quiet_and_interval() -> None:
    clock = _Clock()
    g = idle_gate(clock, snapshot_interval_seconds=300, quiet_seconds=120)
    assert not g.snapshot_due(last_motion=clock.t - 60), "PIR fired a minute ago"
    assert g.snapshot_due(last_motion=clock.t - 121)
    g.observe_idle(None)  # a failed grab still waits out the interval
    clock.t += 200
    assert not g.snapshot_due(last_motion=0.0)
    clock.t += 100
    assert g.snapshot_due(last_motion=0.0)


def test_full_resolution_snapshot_matches_a_sampled_frame() -> None:
    """Camera arrays are shrunk to the size load_frame gives a sampled JPEG."""
    g = idle_gate()
    big = np.kron(scene(), np.ones((4, 4, 1), dtype=np.float32))  # 1280x720
    for _ in range(6):
        g.observe_idle(big)
    assert g.score(scene()) is not None
    assert g.is_empty(scene())


def test_shape_mismatch_warns_once(caplog) -> None:
    g = idle_gate()
    for _ in range(6):
        g.observe_idle(scene())
    with caplog.at_level("WARNING"):
        assert not g.is_empty(scene(w=300, h=300))
        assert not g.is_empty(scene(w=300, h=300))
    assert caplog.text.count("empty gate disabled") == 1


def test_seed_requires_a_visible_port() -> None:
    with pytest.raises(ValueError, match="port"):
        idle_gate().seed(np.zeros((180, 320, 3)))
