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


def run(readings_cm, detection_range_mm=500):
    sensor = FakeVL53L1X(readings_cm)
    events = []
    src = ToFMotionSource(
        detection_range_mm=detection_range_mm, warmup_seconds=0, sensor_factory=lambda: sensor
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
    events, _ = run([80.0, 30.0, 25.0, 20.0, 90.0, 30.0])
    assert len(events) == 2
    assert all(e.trigger is Trigger.TOF for e in events)


def test_invalid_reading_counts_as_clear():
    """None (no valid target) re-arms, so the next bird is not missed."""
    events, _ = run([30.0, None, 30.0])
    assert len(events) == 2


def test_start_and_stop_drive_ranging():
    sensor = FakeVL53L1X([])
    src = ToFMotionSource(warmup_seconds=0, read_rate_hz=100, sensor_factory=lambda: sensor)
    src.start(lambda e: None)
    assert sensor.ranging and sensor.distance_mode == ToFMotionSource.SHORT_DISTANCE_MODE
    src.stop()
    assert not sensor.ranging
