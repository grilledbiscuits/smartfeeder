"""Picamera2Recorder focus control against a stub camera.

Camera Module 3 is an autofocus module and picamera2 leaves imx708 at
LensPosition 1.0 -- one metre -- unless something sets it. Nothing did until
2026-10-09, so every clip was focused about nine times past the perch.
"""

from __future__ import annotations

from capture.recorder import Picamera2Recorder


class _FocusPicam:
    """Records set_controls calls; no camera needed."""

    def __init__(self):
        self.controls = []

    def set_controls(self, d):
        self.controls.append(d)


def test_lens_position_pins_manual_focus():
    """Camera Module 3 defaults to 1.0 dioptres (1 m); the perch is at ~0.11 m."""
    r = Picamera2Recorder(1280, 720, 25, 4000, lens_position=8.8)
    picam = _FocusPicam()
    r._apply_focus(picam)
    assert len(picam.controls) == 1
    sent = picam.controls[0]
    assert sent["LensPosition"] == 8.8
    assert "AfMode" in sent, "must pin AfMode to Manual, not just the position"


def test_no_lens_position_leaves_the_sensor_alone():
    r = Picamera2Recorder(1280, 720, 25, 4000)
    picam = _FocusPicam()
    r._apply_focus(picam)
    assert picam.controls == []


def test_a_lens_less_sensor_only_warns(caplog):
    """Camera Module 2 and the HQ cam have no focus motor; recording must go on."""
    import logging

    class NoLens:
        def set_controls(self, d):
            raise RuntimeError("control AfMode not available")

    r = Picamera2Recorder(1280, 720, 25, 4000, lens_position=8.8)
    with caplog.at_level(logging.WARNING, logger="capture.recorder"):
        r._apply_focus(NoLens())
    assert any("lens_position" in rec.message for rec in caplog.records)
