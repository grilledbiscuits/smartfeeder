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


def run(readings_cm, detection_range_mm=500, release_seconds=0.3, read_rate_hz=10.0):
    """Poll a fixed list of readings. Default release is 3 polls (0.3s at 10 Hz)."""
    sensor = FakeVL53L1X(readings_cm)
    events = []
    src = ToFMotionSource(
        detection_range_mm=detection_range_mm,
        warmup_seconds=0,
        read_rate_hz=read_rate_hz,
        release_seconds=release_seconds,
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
