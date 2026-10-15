#!/usr/bin/env python3
"""Battery workload: record and classify a fresh clip every 30 seconds.

Clips and decoded frames use the Pi's storage and are deleted after each event.
Event and status logs are fsynced for recovery after power loss. No GPIO is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ml" / "src")]


def run_one(recorder, classifier, event_id: str, clip_seconds: float, scratch: Path) -> dict:
    """Run both hardware stages; retain no video or sampled frames."""
    with tempfile.TemporaryDirectory(prefix="smartfeeder-battery-", dir=scratch) as name:
        work = Path(name)
        t = time.monotonic()
        recording = recorder.record(work / f"{event_id}.mp4", clip_seconds)
        recorded_s = time.monotonic() - t
        result = classifier.classify(recording.path, work, event_id)
        classified_s = time.monotonic() - t - recorded_s
        if result.frames_scored < 1 or result.decision is None:
            raise RuntimeError("camera clip produced no classifier decision")
        decision = result.decision
        return {
            "record_s": round(recorded_s, 3),
            "classify_s": round(classified_s, 3),
            "clip_bytes": recording.size_bytes,
            "frames_scored": result.frames_scored,
            "label": getattr(decision, "label", None),
            "confidence": round(float(getattr(decision, "confidence", 0)), 4),
            "is_unknown": getattr(decision, "is_unknown", None),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=30)
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("interval must be positive")

    from capture.build import build_clip_classifier, build_recorder, load_birdcam_config
    from capture.config import CaptureConfig
    from capture.events import MotionEvent, Trigger
    from capture.spool import fsync_dir, fsync_file
    from deploy.bench_pi import board_state

    if ROOT == Path("/opt/smartfeeder"):
        ap.error("run in an isolated checkout, not the live deployment")
    cfg = CaptureConfig.load(args.config, root=ROOT)
    clip_seconds = float(cfg.get("capture.clip_seconds"))
    if args.interval <= clip_seconds:
        ap.error("interval must exceed the camera clip duration")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "events.jsonl").exists():
        raise RuntimeError("choose a fresh output directory; do not mix runs")
    scratch = args.out / "scratch"
    scratch.mkdir(exist_ok=True)
    min_free = int(cfg.get("storage.min_free_mb")) * 1024 * 1024

    recorder = build_recorder(cfg)
    classifier = build_clip_classifier(cfg, load_birdcam_config(ROOT))
    if classifier is None:
        raise RuntimeError("classifier is disabled in the capture config")
    classifier.empty_gate = None  # desk scene must still exercise ONNX inference

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = {
        "started": time.time(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_uptime_s": float(Path("/proc/uptime").read_text().split()[0]),
        "interval_s": args.interval,
        "clip_seconds": clip_seconds,
        "config_sha256": digest(args.config),
        "onnx_sha256": digest(cfg.resolve_path("classifier.onnx_path")),
        "runner_sha256": digest(Path(__file__)),
        "scratch": str(scratch),
        "note": "Desk scene; labels are workload evidence, not accuracy evidence.",
    }
    counts = {"events": 0, "errors": 0}

    def status(state: str, **extra) -> None:
        tmp = args.out / "status.tmp"
        tmp.write_text(json.dumps({**provenance, "state": state, "counts": counts, **extra}, indent=2))
        fsync_file(tmp)
        tmp.replace(args.out / "status.json")
        fsync_dir(args.out)

    def stop_received(_signum, _frame) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("stop requested")

    signal.signal(signal.SIGTERM, stop_received)
    status("running")
    try:
        with (args.out / "events.jsonl").open("a", buffering=1) as log:
            fsync_dir(args.out)
            while True:
                t = time.monotonic()
                event = MotionEvent.now(Trigger.MOCK)
                try:
                    if shutil.disk_usage(scratch).free < min_free:
                        raise RuntimeError("free storage below capture safety floor")
                    result = run_one(recorder, classifier, event.event_id, clip_seconds, scratch)
                    error = None
                except Exception as exc:
                    recorder.close()
                    result = {}
                    error = f"{type(exc).__name__}: {exc}"
                row = {
                    "event_id": event.event_id,
                    "detected_at": event.detected_at.isoformat(timespec="seconds"),
                    "trigger": "mock",
                    **result,
                    "error": error,
                    "seconds": round(time.monotonic() - t, 3),
                    "uptime_s": float(Path("/proc/uptime").read_text().split()[0]),
                    "board": board_state(),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                os.fsync(log.fileno())
                counts["events"] += 1
                counts["errors"] += bool(error)
                status("running", last=row)
                if error:
                    raise RuntimeError(error)
                time.sleep(max(0, args.interval - (time.monotonic() - t)))
    except KeyboardInterrupt:
        status("stopped", finished=time.time())
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())
        raise
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
