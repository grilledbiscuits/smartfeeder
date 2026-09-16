#!/usr/bin/env python3
"""Measure what the classifier actually costs on the target board.

Run this ON THE PI. Nothing here imports birdcam: the Pi venv carries the
capture dependencies only, so this needs numpy and onnxruntime and nothing else.

    scp deploy/bench_pi.py ml/data/export/birdcam_student*.onnx <pi>:/tmp/
    ssh <pi> '/opt/smartfeeder/.venv/bin/python /tmp/bench_pi.py --dir /tmp'

## Why this exists rather than an estimate

`pyproject.toml` says it plainly: latency measured off-target is meaningless.
An x86 laptop and a Cortex-A72 differ in vector width, cache, memory bandwidth
and thermal headroom, and INT8 kernels are exactly where they differ most.

## What it measures, and why each one

* **INT8 against FP32.** On a Pi 4B there is no accelerator, so INT8 with
  XNNPACK's NEON kernels is the shipping configuration and FP32 is the control.
* **XNNPACK against the plain CPU provider.** The capture config asks for
  XNNPACK first and falls back silently if it was not built into the installed
  onnxruntime. That fallback is invisible in the logs and costs real time, so it
  is worth seeing as a number.
* **Sustained, not burst.** A Pi 4B throttles at 80C and drops its clock. A cold
  ten-iteration benchmark measures a state the feeder will never be in during a
  busy hour, so this runs long enough to reach a steady state and reports the
  first and last thirds separately.
* **Per-CLIP cost, not per-frame.** The service classifies `max_frames: 12`
  sampled frames per motion event and votes. Twelve frames is the number that
  has to fit between events.
* **The empty gate.** It now runs before the recorder on every motion event, so
  its cost is on the hot path even when nothing is recorded. It is numpy, not a
  model, and should be negligible -- this checks that it is.

Random input is used for timing. A convolutional graph has no data-dependent
control flow, so activations change the numbers but not the arithmetic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def board_state() -> dict:
    """Temperature, ARM clock and the throttle flags, where available."""
    out: dict = {}
    try:
        t = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        out["temp_c"] = round(int(t) / 1000.0, 1)
    except Exception:
        pass
    for key, cmd in (
        ("arm_hz", ["vcgencmd", "measure_clock", "arm"]),
        ("throttled", ["vcgencmd", "get_throttled"]),
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                out[key] = r.stdout.strip().split("=")[-1]
        except Exception:
            pass
    return out


def describe_throttle(flags: str | None) -> str:
    """Decode vcgencmd's bitfield, because 0x50000 means nothing on its own."""
    if not flags:
        return "unknown"
    try:
        v = int(flags, 16)
    except ValueError:
        return flags
    now = {0: "under-voltage", 1: "arm frequency capped", 2: "currently throttled",
           3: "soft temp limit"}
    ever = {16: "under-voltage occurred", 17: "arm frequency capping occurred",
            18: "throttling occurred", 19: "soft temp limit occurred"}
    hits = [m for b, m in {**now, **ever}.items() if v & (1 << b)]
    return ", ".join(hits) if hits else "clean"


def run_model(path: Path, providers: list[str], size: int, seconds: float, warmup: int = 5):
    """Time single-frame inference for `seconds`, after a warmup."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    wanted = [p for p in providers if p in available]
    if not wanted:
        return {"error": f"none of {providers} available; have {sorted(available)}"}

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4  # the Pi 4B has four cores and nothing else to do
    sess = ort.InferenceSession(str(path), so, providers=wanted)
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, size, size).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {name: x})

    lat: list[float] = []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        lat.append((time.perf_counter() - t0) * 1000.0)

    third = max(1, len(lat) // 3)
    return {
        "providers_used": wanted,
        "fell_back": wanted != list(providers),
        "n": len(lat),
        "median_ms": round(statistics.median(lat), 1),
        "p95_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
        "first_third_median_ms": round(statistics.median(lat[:third]), 1),
        "last_third_median_ms": round(statistics.median(lat[-third:]), 1),
        "clip_12_frames_s": round(statistics.median(lat) * 12 / 1000.0, 2),
    }


def bench_empty_gate(frames: int = 48, side: int = 320, reps: int = 20) -> dict:
    """The rolling-median background and one port check, as the gate does it."""
    h = int(side * 9 / 16)
    buf = [np.random.rand(side, h, 3).astype(np.float32) * 255 for _ in range(frames)]
    med, chk = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        bg = np.median(np.stack(buf), axis=0)
        med.append((time.perf_counter() - t0) * 1000.0)
        box = bg[40:140, 40:140]
        t0 = time.perf_counter()
        float(np.abs(buf[0][40:140, 40:140] - box).mean())
        chk.append((time.perf_counter() - t0) * 1000.0)
    return {
        "background_median_ms": round(statistics.median(med), 1),
        "port_check_ms": round(statistics.median(chk), 3),
        "note": "background is recomputed only when the buffer changes, not per check",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("/tmp"), help="where the .onnx files are")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seconds", type=float, default=60.0, help="per configuration")
    ap.add_argument("--out", type=Path, default=Path("/tmp/bench_pi.json"))
    args = ap.parse_args()

    print(f"board before: {board_state()}")
    results: dict = {"before": board_state(), "size": args.size, "models": {}}

    configs = [
        ("int8_xnnpack", "birdcam_student_int8.onnx",
         ["XnnpackExecutionProvider", "CPUExecutionProvider"]),
        ("int8_cpu", "birdcam_student_int8.onnx", ["CPUExecutionProvider"]),
        ("fp32_xnnpack", "birdcam_student.onnx",
         ["XnnpackExecutionProvider", "CPUExecutionProvider"]),
        ("fp32_cpu", "birdcam_student.onnx", ["CPUExecutionProvider"]),
    ]
    for label, fname, providers in configs:
        path = args.dir / fname
        if not path.is_file():
            print(f"{label:<16} SKIP (no {path})")
            continue
        r = run_model(path, providers, args.size, args.seconds)
        r["state_after"] = board_state()
        results["models"][label] = r
        if "error" in r:
            print(f"{label:<16} {r['error']}")
            continue
        drift = r["last_third_median_ms"] - r["first_third_median_ms"]
        print(
            f"{label:<16} median {r['median_ms']:>7.1f} ms   p95 {r['p95_ms']:>7.1f}   "
            f"clip(12) {r['clip_12_frames_s']:>5.2f} s   drift {drift:+.1f} ms"
            f"{'   [PROVIDER FELL BACK]' if r['fell_back'] else ''}"
        )

    results["empty_gate"] = bench_empty_gate()
    print(f"\nempty gate: {results['empty_gate']}")

    after = board_state()
    results["after"] = after
    print(f"\nboard after: {after}")
    print(f"throttle: {describe_throttle(after.get('throttled'))}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
