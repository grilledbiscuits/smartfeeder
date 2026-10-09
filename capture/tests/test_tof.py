"""ToF motion source against a fake VL53L1X: units, edges and invalid readings."""

from __future__ import annotations

from capture.events import Trigger
from capture.motion import ToFMotionSource


class FakeVL53L1X:
    """Mimics adafruit_vl53l1x: distance in cm, None when invalid, one reading per clear."""

    def __init__(self, readings_cm):
        self.readings = list(readings_cm)
        self.ranging = False
        self.distance_mode = None
        self._pending = False

    def start_ranging(self):
        self.ranging = True

    def stop_ranging(self):
        self.ranging = False

    @property
    def data_ready(self):
        if not self._pending and self.readings:
            self._pending = True
        return self._pending

    @property
    def distance(self):
        return self.readings[0]

    def clear_interrupt(self):
        self.readings.pop(0)
        self._pending = False


def run(
    readings_cm,
    detection_range_mm=500,
    release_seconds=0.3,
    read_rate_hz=10.0,
    max_hold_seconds=None,
):
    """Poll a fixed list of readings. Default release is 3 polls (0.3s at 10 Hz)."""
    sensor = FakeVL53L1X(readings_cm)
    events = []
    src = ToFMotionSource(
        detection_range_mm=detection_range_mm,
        warmup_seconds=0,
        read_rate_hz=read_rate_hz,
        release_seconds=release_seconds,
        max_hold_seconds=max_hold_seconds,
        sensor_factory=lambda: sensor,
    )
    src._sensor = src._sensor_factory()  # bypass the polling thread
    src._callback = events.append
    while sensor.readings:
        src.poll_once()
    return events, sensor


def test_distance_is_compared_in_millimetres():
    """120 cm is 1200 mm: outside a 500 mm range. The old code fired on it."""
    events, _ = run([120.0, 80.0, 60.0])
    assert events == []


def test_fires_once_on_entering_range_then_rearms():
    """Three clear reads release the trigger, so the second arrival fires again."""
    events, _ = run([80.0, 30.0, 25.0, 20.0, 90.0, 90.0, 90.0, 30.0])
    assert len(events) == 2
    assert all(e.trigger is Trigger.TOF for e in events)


def test_a_dropout_mid_visit_does_not_refire():
    """The bug this hysteresis exists for: one presence, one event.

    MEASURED 2026-10-06 -- about half the sensor's reads come back invalid, so
    without damping this pattern fired an event per surviving read.
    """
    events, _ = run([30.0, None, 30.0, None, None, 28.0, None, 25.0])
    assert len(events) == 1


def test_sustained_invalid_readings_do_release():
    """An empty port is reported as None, so None must still re-arm eventually."""
    events, _ = run([30.0, None, None, None, 30.0])
    assert len(events) == 2


def test_release_is_counted_in_polls_not_readings():
    """release_seconds converts via read_rate_hz; 0.5s at 20 Hz is 10 polls."""
    src = ToFMotionSource(release_seconds=0.5, read_rate_hz=20.0)
    assert src._release_polls == 10
    # Never rounds to zero, which would restore the old fire-on-every-read bug.
    assert ToFMotionSource(release_seconds=0.0, read_rate_hz=10.0)._release_polls == 1


def test_start_and_stop_drive_ranging():
    sensor = FakeVL53L1X([])
    src = ToFMotionSource(warmup_seconds=0, read_rate_hz=100, sensor_factory=lambda: sensor)
    src.start(lambda e: None)
    assert sensor.ranging and sensor.distance_mode == ToFMotionSource.SHORT_DISTANCE_MODE
    src.stop()
    assert not sensor.ranging


def test_a_stuck_gate_releases_and_warns(caplog):
    """Something parked in the beam must not silently blind the feeder.

    The perch sits ~20 cm from the sensor, inside any sensible detection range,
    so a sensor aimed slightly differently could read it forever. Without the
    valve the trigger latches and the log says nothing at all.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="capture.motion"):
        # 10 Hz, 1s hold -> 10 polls; the 11th in-range read trips the valve.
        events, _ = run([20.0] * 25, release_seconds=0.3, max_hold_seconds=1.0)
    assert len(events) > 1, "a permanently held gate must re-fire, not go quiet"
    assert any("stuck" in r.message for r in caplog.records)


def test_the_valve_can_be_disabled():
    events, _ = run([20.0] * 40, release_seconds=0.3, max_hold_seconds=None)
    assert len(events) == 1


def test_the_valve_does_not_disturb_a_normal_visit():
    """A visit well under max_hold_seconds stays exactly one event."""
    events, _ = run([None] + [20.0] * 15 + [None] * 5, release_seconds=0.3, max_hold_seconds=60.0)
    assert len(events) == 1


class RoiSensor(FakeVL53L1X):
    """Records ROI writes and when they happened relative to start_ranging()."""

    def __init__(self, readings_cm=()):
        super().__init__(readings_cm)
        self.roi_xy = (16, 16)
        self.roi_center = 199
        self.writes_after_start = 0

    def start_ranging(self):
        super().start_ranging()
        self._started = True

    def __setattr__(self, name, value):
        if name in ("roi_xy", "roi_center") and getattr(self, "_started", False):
            object.__setattr__(self, "writes_after_start", self.writes_after_start + 1)
        object.__setattr__(self, name, value)


def test_roi_is_applied_before_ranging_starts():
    """Order matters: an ROI set mid-ranging gives readings that do not reproduce."""
    sensor = RoiSensor()
    src = ToFMotionSource(
        warmup_seconds=0,
        read_rate_hz=100,
        roi_size=4,
        roi_center=199,
        sensor_factory=lambda: sensor,
    )
    src.start(lambda e: None)
    src.stop()
    assert sensor.roi_xy == (4, 4)
    assert sensor.roi_center == 199
    assert sensor.writes_after_start == 0, "ROI must be set before start_ranging()"


def test_roi_defaults_leave_the_sensor_untouched():
    sensor = RoiSensor()
    src = ToFMotionSource(warmup_seconds=0, read_rate_hz=100, sensor_factory=lambda: sensor)
    src.start(lambda e: None)
    src.stop()
    assert sensor.roi_xy == (16, 16), "no roi_size configured -> full array"


def test_a_driver_without_roi_support_only_warns(caplog):
    """Older drivers expose no ROI; that must not stop the service starting."""
    import logging

    class NoRoi(FakeVL53L1X):
        __slots__ = ()

        def __setattr__(self, name, value):
            if name in ("roi_xy", "roi_center"):
                raise AttributeError(name)
            object.__setattr__(self, name, value)

    sensor = NoRoi([])
    src = ToFMotionSource(
        warmup_seconds=0,
        read_rate_hz=100,
        roi_size=4,
        roi_center=199,
        sensor_factory=lambda: sensor,
    )
    with caplog.at_level(logging.WARNING, logger="capture.motion"):
        src.start(lambda e: None)
    src.stop()
    assert any("no ROI control" in r.message for r in caplog.records)


def test_out_of_range_roi_values_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="roi_size"):
        ToFMotionSource(roi_size=2)
    with pytest.raises(ValueError, match="roi_center"):
        ToFMotionSource(roi_center=999)


# --- baseline-relative detection ---------------------------------------------
#
# The model that replaces the absolute range. The perch is permanently in the
# beam, so "within 500 mm" is always true and the gate never produces a rising
# edge; these tests pin the behaviour that fixes that.


def run_baseline(readings_cm, margin_mm=20, samples=5, release_seconds=0.3):
    sensor = FakeVL53L1X(readings_cm)
    events = []
    src = ToFMotionSource(
        warmup_seconds=0,
        read_rate_hz=10.0,
        release_seconds=release_seconds,
        baseline_margin_mm=margin_mm,
        baseline_samples=samples,
        sensor_factory=lambda: sensor,
    )
    src._sensor = src._sensor_factory()
    src._callback = events.append
    while sensor.readings:
        src.poll_once()
    return events, src


def test_perch_in_the_beam_fires_nothing():
    """The whole fault: a static object at 13 cm must not look like a visit."""
    events, src = run_baseline([13.0] * 40)
    assert events == []
    assert src._baseline == 130.0


def test_absolute_range_fires_on_the_static_perch():
    """Contrast case, documenting why the absolute model had to go."""
    events, _ = run([13.0] * 40, detection_range_mm=500)
    assert len(events) == 1  # ...and then never again, however many birds land


def test_bird_nearer_than_the_perch_fires_once():
    events, _ = run_baseline([13.0] * 10 + [10.5] * 20)
    assert len(events) == 1


def test_departure_smaller_than_the_margin_does_not_fire():
    events, _ = run_baseline([13.0] * 10 + [12.2] * 20, margin_mm=20)
    assert events == []


def test_static_object_leaving_also_fires():
    """Symmetric margin: farther is a change too, and we would rather record it."""
    events, _ = run_baseline([13.0] * 10 + [25.0] * 20)
    assert len(events) == 1


def test_bird_cannot_teach_the_baseline_that_it_is_furniture():
    """A long dwell must stay detected, not get absorbed into the resting median."""
    events, src = run_baseline([13.0] * 10 + [10.0] * 60, samples=5)
    assert len(events) == 1
    assert src._baseline == 130.0
    assert src._in_range


def test_relearns_after_the_mount_settles_to_a_new_distance():
    events, src = run_baseline([13.0] * 10 + [10.0] * 10 + [10.0] * 10, samples=5)
    # Still one event; the new distance never clears, so the valve (not the
    # baseline) is what recovers this. Recorded so the ceiling is explicit.
    assert len(events) == 1


def test_returns_beyond_the_scene_neither_fire_nor_move_the_baseline():
    events, src = run_baseline([13.0] * 10 + [120.0] * 10 + [13.0] * 10)
    assert events == []
    assert src._baseline == 130.0


def test_dropouts_during_a_visit_do_not_retrigger():
    events, _ = run_baseline([13.0] * 10 + [10.0, None, 10.0, None, None, 10.0] * 5)
    assert len(events) == 1


def test_no_detection_until_the_baseline_is_learned():
    """Before `baseline_samples` clear reads, nothing fires -- including the
    very first reading, which otherwise has nothing to be compared against."""
    events, src = run_baseline([10.0] * 4, samples=5)
    assert events == []
    assert src._baseline is None
