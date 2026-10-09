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

MARGINS_MM = (5, 10, 15, 20, 30)


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


def report(trace: list[float | None], samples: int) -> int:
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
    sys.path.insert(0, "/opt/smartfeeder")
    from capture.motion import ToFMotionSource

    print(f"\nempty-port trigger rate by margin ({len(trace)} polls replayed):")
    print("  margin   events   per minute")
    best = None
    for margin in MARGINS_MM:
        src = ToFMotionSource(
            warmup_seconds=0,
            baseline_margin_mm=margin,
            baseline_samples=samples,
            sensor_factory=lambda: None,
        )
        events = 0
        for mm in trace:
            if src._detect(mm):
                if not src._in_range:
                    events += 1
                src._in_range = True
                src._clear_run = 0
            else:
                src._clear_run += 1
                if src._clear_run >= src._release_polls:
                    src._in_range = False
        per_min = events / (len(trace) / 10.0) * 60
        print(f"  {margin:3d} mm   {events:6d}   {per_min:8.1f}")
        if best is None and events == 0:
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
    args = ap.parse_args()

    print(
        f"Ranging for {args.seconds:.0f}s at {args.rate_hz:.0f} Hz, "
        f"ROI {args.roi_size}x{args.roi_size} centre {args.roi_center}"
    )
    if args.note:
        print(f"note: {args.note}")
    trace = read_trace(args.seconds, args.rate_hz, args.roi_size, args.roi_center)
    return report(trace, args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
