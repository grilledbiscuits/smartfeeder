#!/usr/bin/env python3
"""Bounded Pi-only replay soak. Repeated clips test stability, not new accuracy.

Uses one pipeline throughout to expose accumulating memory/state errors. Writes
one flushed JSONL row per event and atomically replaces status.json. No camera
or GPIO is opened. Run in a separate staging checkout; never the live root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ml" / "src")]


def main():
    from capture.build import build_service
    from capture.config import CaptureConfig
    from capture.events import MotionEvent, Trigger
    from deploy.bench_pi import board_state
    from deploy.replay_sequence import is_correct
    from web import paths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cycles", type=int, default=72)
    ap.add_argument("--interval", type=float, default=30)
    args = ap.parse_args()
    if args.cycles < 1 or args.interval < 0:
        ap.error("cycles must be positive and interval nonnegative")
    if ROOT == Path("/opt/smartfeeder") or not paths.DB_PATH.is_relative_to(ROOT):
        raise RuntimeError("soak must use an isolated checkout and database")
    cfg = CaptureConfig.load(args.config, root=ROOT)
    for key in ("work_dir", "pending_dir", "review_dir"):
        if not cfg.resolve_path(f"storage.{key}").is_relative_to(ROOT):
            raise RuntimeError("soak storage points outside staging root")
    manifest = json.loads((args.clips / "manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "events.jsonl").exists():
        raise RuntimeError("choose a fresh output directory; do not mix runs")
    _svc, pipe = build_service(cfg, mock=True, replay=args.clips / manifest[0]["file"])
    targets = pipe.classifier.decider._capture_targets
    counts = Counter()
    started = time.time()
    model = cfg.resolve_path("classifier.onnx_path")

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = dict(
        pid=os.getpid(),
        root=str(ROOT),
        started=started,
        onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        calibration_sha256=file_hash(cfg.resolve_path("classifier.operating_points")),
        taxonomy_sha256=file_hash(ROOT / "ml/config/taxonomy.yaml"),
        manifest_sha256=file_hash(args.clips / "manifest.json"),
        runner_sha256=file_hash(Path(__file__)),
        caveat="Repeated replay stability test; not independent accuracy evidence.",
    )

    def status(state, **fields):
        value = dict(provenance, state=state, counts=dict(counts), **fields)
        temp = args.out / "status.tmp"
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(args.out / "status.json")

    status("running")

    def stop_received(_signum, _frame):
        raise KeyboardInterrupt("stop requested")

    signal.signal(signal.SIGTERM, stop_received)
    try:
        with (args.out / "events.jsonl").open("a", buffering=1) as log:
            for cycle in range(args.cycles):
                for entry in manifest:
                    t = time.monotonic()
                    pipe.recorder.source = args.clips / entry["file"]
                    rec = pipe.handle(MotionEvent.now(trigger=Trigger.MOCK))
                    correct, why = is_correct(entry["truth"], rec)
                    expected_publish = entry["truth"] in targets
                    published = bool(rec.outcome and rec.outcome.value == "publish")
                    action_correct = published == expected_publish and not rec.error
                    rss = next(
                        (
                            line.split()[1]
                            for line in Path("/proc/self/status").read_text().splitlines()
                            if line.startswith("VmRSS:")
                        ),
                        None,
                    )
                    row = dict(
                        cycle=cycle,
                        file=entry["file"],
                        truth=entry["truth"],
                        **rec.log_fields(),
                        correct=correct,
                        why=why,
                        action_correct=action_correct,
                        rss_kb=int(rss) if rss else None,
                        seconds=time.monotonic() - t,
                        board=board_state(),
                    )
                    log.write(json.dumps(row) + "\n")
                    counts["events"] += 1
                    counts["errors"] += bool(rec.error)
                    counts["identification_correct"] += correct
                    counts["action_correct"] += action_correct
                    status("running", last=row)
                    if rec.error:
                        raise RuntimeError(rec.error)
                    if row["rss_kb"] and row["rss_kb"] > 512 * 1024:
                        raise RuntimeError("soak RSS exceeded the 512 MiB safety limit")
                    # Stop on current voltage/thermal/capping flags; historical
                    # sticky flags are recorded but do not abort the experiment.
                    flags = row["board"].get("throttled")
                    if flags and int(flags, 16) & 0xF:
                        raise RuntimeError(f"current board warning flags: {flags}")
                    time.sleep(max(0, args.interval - (time.monotonic() - t)))
        status("complete", finished=time.time(), board=board_state())
    except KeyboardInterrupt:
        status("stopped", finished=time.time())
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())
        raise
    finally:
        pipe.recorder.close()


if __name__ == "__main__":
    main()
