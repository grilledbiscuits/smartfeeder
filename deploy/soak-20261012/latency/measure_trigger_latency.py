#!/usr/bin/env python3
"""Measure software-trigger admission to the first encoded camera frame on a Pi.

Run after the replay soak. The ToF is not read; a mock event exercises the
deployed pipeline and real camera, with results kept in this isolated checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ml" / "src")]


def main() -> None:
    from picamera2 import outputs

    from capture.build import build_service
    from capture.config import CaptureConfig
    from capture.events import MotionEvent, Trigger
    from capture.recorder import Picamera2Recorder

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()
    if args.trials < 2:
        ap.error("at least two trials are needed for cold and warm camera starts")
    if ROOT == Path("/opt/smartfeeder"):
        ap.error("use an isolated checkout, not the live deployment")
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "events.jsonl"
    if rows_path.exists():
        ap.error("choose a fresh output directory")

    class TimedOutput(outputs.FfmpegOutput):
        arrivals_ns: list[int] = []

        def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
            super().outputframe(frame, keyframe, timestamp, packet, audio)
            if frame and self.recording and not self.output_broken and not getattr(self, "_seen", False):
                self._seen = True
                self.arrivals_ns.append(time.monotonic_ns())

    outputs.FfmpegOutput = TimedOutput
    cfg = CaptureConfig.load(args.config, root=ROOT)
    _service, pipe = build_service(cfg, mock=True)
    recorder = pipe.recorder
    if not isinstance(recorder, Picamera2Recorder):
        raise RuntimeError("latency test must use the real Pi camera recorder")

    provenance = {
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": "admitted software trigger to first encoded frame delivered to FFmpeg",
        "limitation": "ToF sensor detection, GPIO callback, and queue delay are excluded",
    }
    rows = []
    try:
        with rows_path.open("w", buffering=1) as log:
            for trial in range(args.trials):
                cold = trial % 2 == 0
                if cold:
                    recorder.close()
                before = len(TimedOutput.arrivals_ns)
                trigger_ns = time.monotonic_ns()
                event = MotionEvent.now(Trigger.MOCK)
                result = pipe.handle(event)
                if len(TimedOutput.arrivals_ns) != before + 1 or result.duration_seconds is None:
                    raise RuntimeError(f"trial {trial}: no valid first frame ({result.error})")
                latency_ms = (TimedOutput.arrivals_ns[-1] - trigger_ns) / 1_000_000
                assert 0 <= latency_ms < 60_000, f"invalid latency: {latency_ms} ms"
                row = {
                    "trial": trial,
                    "camera": "cold" if cold else "warm",
                    "event_id": event.event_id,
                    "trigger_at": event.detected_at.isoformat(),
                    "first_frame_latency_ms": round(latency_ms, 1),
                    "record_duration_s": result.duration_seconds,
                    "error": result.error,
                }
                log.write(json.dumps(row) + "\n")
                rows.append(row)
    finally:
        recorder.close()

    by_mode = {
        mode: {
            "n": len(values),
            "median_ms": round(statistics.median(values), 1),
            "max_ms": round(max(values), 1),
            "within_2s": sum(value <= 2000 for value in values),
        }
        for mode in ("cold", "warm")
        if (values := [row["first_frame_latency_ms"] for row in rows if row["camera"] == mode])
    }
    summary = {**provenance, "trials": len(rows), "results": by_mode}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
