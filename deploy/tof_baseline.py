#!/usr/bin/env python3
"""Measure the ToF resting baseline at the feeder and fit motion.baseline_margin_mm.

Run this on the Pi, with the feeder EMPTY and the perch in its normal position,
with the capture service stopped (both cannot hold the sensor):

    sudo systemctl stop birdcam-capture
    sudo /opt/smartfeeder/.venv/bin/python deploy/tof_baseline.py --seconds 60
    sudo systemctl start birdcam-capture

What it reports, and why each part matters:

- the resting distribution: median is what the source learns as its baseline,
  and the spread is the noise the margin has to clear.
- the false-trigger rate each candidate margin would produce ON AN EMPTY PORT,
  replayed against the readings just taken. Pick the smallest margin whose rate
  you can live with; smaller is safer, because an empty-port trigger costs a
  discarded clip while a missed visit is unrecoverable and logs nothing.

This fixes the lower bound only. The upper bound -- the largest margin that
still catches a perched sunbird -- cannot be measured without a bird on the
perch, so run with `--note` while watching the feeder and keep the trace.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

MARGINS_MM = (5, 7, 8, 10, 15, 20, 30)


def read_trace(seconds: float, rate_hz: float, roi_size: int | None, roi_center: int | None):
    import adafruit_vl53l1x
    import board

    sensor = adafruit_vl53l1x.VL53L1X(board.I2C())
    sensor.distance_mode = 1  # SHORT, as the service uses
    # Order matters: an ROI set while ranging is live gives readings that do not
    # reproduce (see capture.motion.ToFMotionSource._apply_roi).
    if roi_size is not None:
        sensor.roi_xy = (roi_size, roi_size)
    if roi_center is not None:
        sensor.roi_center = roi_center
    sensor.start_ranging()
    try:
        trace: list[float | None] = []
        interval = 1.0 / rate_hz
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if sensor.data_ready:
                cm = sensor.distance
                sensor.clear_interrupt()
                trace.append(None if cm is None else cm * 10.0)
            time.sleep(interval)
        return trace
    finally:
        sensor.stop_ranging()


class _TraceSensor:
    """A VL53L1X that plays back a recorded trace, for the replay above.

    Mirrors the driver's contract exactly, because poll_once depends on it:
    `distance` is in CENTIMETRES, None means an invalid read, and a new reading
    only appears after `clear_interrupt()`.
    """

    def __init__(self, trace: list[float | None]) -> None:
        self._trace = list(trace)

    @property
    def remaining(self) -> int:
        return len(self._trace)

    @property
    def data_ready(self) -> bool:
        return bool(self._trace)

    @property
    def distance(self) -> float | None:
        mm = self._trace[0]
        return None if mm is None else mm / 10.0

    def clear_interrupt(self) -> None:
        self._trace.pop(0)


def report(
    trace: list[float | None],
    samples: int,
    rate_hz: float = 10.0,
    *,
    release_seconds: float = 1.0,
    max_hold_seconds: float | None = 60.0,
) -> int:
    valid = [mm for mm in trace if mm is not None]
    pct = 100 * len(valid) / max(1, len(trace))
    print(f"\n{len(trace)} polls, {len(valid)} valid ({pct:.0f}%)")
    if len(valid) < samples:
        print(
            f"NOT ENOUGH VALID READS to fit anything (need {samples}, the "
            f"baseline_samples window). The sensor is not seeing the scene: "
            f"check the aim, the ROI and that nothing is closer than the perch."
        )
        return 1

    median = statistics.median(valid)
    print(f"resting distance: median {median:.0f} mm ({median / 10:.1f} cm)")
    print(f"  stdev {statistics.pstdev(valid):.1f} mm, min {min(valid):.0f}, max {max(valid):.0f}")
    qs = statistics.quantiles(valid, n=100)
    print(f"  p1 {qs[0]:.0f} mm, p99 {qs[-1]:.0f} mm")

    # Replay the real source against this trace, at each candidate margin, so
    # the rates come from the same code path that runs in the service.
    #
    # "The same code path" has to mean poll_once, not a reimplementation of it.
    # The first version of this script hand-rolled the state machine here and
    # left out the max_hold_seconds valve, so the replayed gate latched on its
    # first detection and never re-armed -- reporting exactly 1 event for every
    # margin from 5 to 15 mm, which reads as "quiet" and is actually "stuck".
    # Both the 7 mm and the 20 mm margins were fitted on that artifact.
    for candidate in ("/opt/smartfeeder", str(Path(__file__).resolve().parent.parent)):
        if candidate not in sys.path:
            sys.path.append(candidate)
    from capture.motion import ToFMotionSource

    print(f"\nempty-port trigger rate by margin ({len(trace)} polls replayed):")
    print("  margin   events   per minute")
    best = None
    for margin in MARGINS_MM:
        events: list = []
        src = ToFMotionSource(
            warmup_seconds=0,
            read_rate_hz=rate_hz,
            release_seconds=release_seconds,
            max_hold_seconds=max_hold_seconds,
            baseline_margin_mm=margin,
            baseline_samples=samples,
            sensor_factory=lambda: _TraceSensor(trace),
        )
        src._sensor = src._sensor_factory()
        src._callback = events.append
        while src._sensor.remaining:
            src.poll_once()
        per_min = len(events) / (len(trace) / rate_hz) * 60
        print(f"  {margin:3d} mm   {len(events):6d}   {per_min:8.1f}")
        if best is None and not events:
            best = margin

    if best is None:
        print(
            "\nEvery candidate margin fires on an empty port. The resting scene is "
            "not stable enough to subtract: something in the beam is moving "
            "(leaves, the bottle swinging) or the valid-read rate is too low."
        )
        return 1
    print(f"\nSmallest margin with no empty-port trigger: {best} mm")
    print(f"  set motion.baseline_margin_mm: {best}")
    print(
        "  Then confirm it CATCHES a bird: watch the feeder and check a real visit\n"
        "  produces a trigger. This measurement only rules out false triggers."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--roi-size", type=int, default=16)
    ap.add_argument("--roi-center", type=int, default=199)
    ap.add_argument("--samples", type=int, default=30, help="motion.baseline_samples")
    ap.add_argument("--note", default="", help="recorded in the output, e.g. 'bird present'")
    # These MUST match the live capture.yaml: the valve in particular decides
    # whether a latched gate re-arms, and leaving it out is what made the first
    # two margin fits meaningless.
    ap.add_argument("--release-seconds", type=float, default=1.0)
    ap.add_argument("--max-hold-seconds", type=float, default=60.0)
    ap.add_argument("--save-trace", default="", help="write the raw mm readings here")
    args = ap.parse_args()

    print(
        f"Ranging for {args.seconds:.0f}s at {args.rate_hz:.0f} Hz, "
        f"ROI {args.roi_size}x{args.roi_size} centre {args.roi_center}"
    )
    if args.note:
        print(f"note: {args.note}")
    trace = read_trace(args.seconds, args.rate_hz, args.roi_size, args.roi_center)
    if args.save_trace:
        Path(args.save_trace).write_text(
            "\n".join("" if mm is None else f"{mm:.0f}" for mm in trace) + "\n"
        )
        print(f"raw trace saved to {args.save_trace} ({len(trace)} polls)")
    return report(
        trace,
        args.samples,
        args.rate_hz,
        release_seconds=args.release_seconds,
        max_hold_seconds=args.max_hold_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
