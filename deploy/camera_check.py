#!/usr/bin/env python3
"""Exercise the camera paths that only real hardware can exercise.

    sudo systemctl stop birdcam-capture     # it holds the camera
    python deploy/camera_check.py
    sudo systemctl start birdcam-capture

`peek()` and the empty gate ahead of the recorder were written against fakes.
Fakes cannot answer the questions that matter here:

* Does `capture_array` return frames of the shape and dtype the gate expects?
* **Can `record()` start after `peek()` has already started the camera?**
  `peek()` calls `picam.start()` to pull a still; `record()` then calls
  `start_recording()` on a camera that is already running. Whether picamera2
  accepts that ordering is exactly the kind of integration fault that passes
  every unit test and fails on the first real motion event.
* Does a second `peek()` after a recording still work, since the pipeline peeks
  on every event?

Privacy: frames are reduced to statistics on the device and never written out.
The one test recording is checked for validity and deleted.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ml" / "src")]


def stats(frames) -> dict:
    import numpy as np

    if not frames:
        return {"n": 0}
    a = frames[0]
    return {
        "n": len(frames),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "mean": round(float(np.mean(a)), 1),
        "min": round(float(np.min(a)), 1),
        "max": round(float(np.max(a)), 1),
        "frames_identical": bool(len(frames) > 1 and np.array_equal(frames[0], frames[-1])),
    }


def main() -> None:
    from capture.emptygate import EmptyGate, find_port
    from capture.recorder import Picamera2Recorder

    out: dict = {}
    rec = Picamera2Recorder(width=1280, height=720, framerate=25, bitrate_kbps=4000)
    try:
        t0 = time.perf_counter()
        f1 = rec.peek(count=3, interval_seconds=0.25)
        out["peek_1"] = {**stats(f1), "seconds": round(time.perf_counter() - t0, 2)}

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "check.mp4"
            t0 = time.perf_counter()
            try:
                r = rec.record(dest, seconds=3.0)
                out["record_after_peek"] = {
                    "ok": True,
                    "size_bytes": r.size_bytes,
                    "duration_s": r.duration_seconds,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            except Exception as exc:  # noqa: BLE001 - this is the question being asked
                out["record_after_peek"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            # dest is removed with the temporary directory

        t0 = time.perf_counter()
        f2 = rec.peek(count=3, interval_seconds=0.25)
        out["peek_after_record"] = {**stats(f2), "seconds": round(time.perf_counter() - t0, 2)}

        frames = f1 + f2
        if frames:
            import numpy as np

            from capture.emptygate import load_frame  # noqa: F401  (import check only)

            small = [
                np.asarray(f[:: max(1, f.shape[0] // 180), :: max(1, f.shape[1] // 320)])
                for f in frames
            ]
            g = EmptyGate()
            for _ in range(4):
                for s in small:
                    g.observe(s)
            bg = g.background()
            out["gate"] = {
                "ready": g.ready,
                "port_found_in_scene": find_port(np.median(np.stack(small), axis=0)) is not None,
                "background_ok": bg is not None,
                "note": "a desk or wall has no red feeder port, so port_found false is expected "
                "and the gate should fail OPEN (never report empty)",
                "is_empty_on_live_frame": g.is_empty(small[0]),
            }
    finally:
        rec.close()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
