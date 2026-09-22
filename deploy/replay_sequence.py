#!/usr/bin/env python3
"""Drive the real capture pipeline over a sequence of real clips, on the target.

    python deploy/replay_sequence.py --config deploy/replay.yaml --clips replay/

This is the closest thing to a live test that needs neither a feeder nor a bird.
Every stage is the production one -- recorder, empty gate, INT8 classifier,
novelty gate, rollup, vote, keep/discard decision, publication -- and only the
camera is replaced, by a recorder that copies a clip instead of filming.

## Scope

Reuses one pipeline across clips to exercise persistent state. Production uses
an explicitly supplied empty reference; the reference clip must be excluded
from reported evaluation counts. An empty clip succeeds when discarded, whether
screening ran before or after recording. Identification success on a bird does
not by itself establish correct publication policy; soak_replay also checks
whether the truth belongs to the capture allowlist.

Prefer clips with traceable split/image provenance. Repeatedly examined test
clips are development evidence, not an untouched field holdout. Replay bypasses
PIR admission and camera timing; it cannot test missed arrivals in hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ml" / "src")]


def _genus_of(slug: str) -> str:
    return slug.split("_", 1)[0]


def _family_of(slug: str) -> str | None:
    from birdcam.config import load_config

    cfg = load_config()
    genus = next((s.genus for s in cfg.species if s.slug == slug), None)
    return cfg.genus_to_family.get(genus) if genus else None


def is_correct(truth: str, record) -> tuple[bool, str]:
    from capture.events import Outcome

    if record.error:
        return False, f"pipeline error: {record.error}"
    d = record.decision
    recorded = record.clip_path is not None
    if truth == "empty_feeder":
        discarded = record.outcome == Outcome.DISCARD
        return discarded, "discarded empty clip" if discarded else "empty clip kept"
    if record.empty:
        return False, "occupied clip rejected by empty gate"
    if not recorded:
        return (False, "bird not recorded")
    if d is None:
        return (False, "recorded but not classified")
    label = getattr(d, "label", "")
    if label == truth:
        return (True, "exact")
    if label.endswith("_indet") and label.startswith(_genus_of(truth)):
        return (True, "genus rollup")
    if getattr(d, "level", "") == "family":
        family = _family_of(truth)
        if family and label == f"{family.lower()}_indet":
            return (True, "family rollup")
    return (False, f"called it {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True, help="directory with manifest.json")
    ap.add_argument("--out", type=Path, default=Path("replay_results.json"))
    args = ap.parse_args()

    from capture.build import build_service
    from capture.config import CaptureConfig
    from capture.events import MotionEvent, Trigger

    manifest = json.loads((args.clips / "manifest.json").read_text())
    cfg = CaptureConfig.load(args.config, root=ROOT)
    first = args.clips / manifest[0]["file"]
    _service, pipeline = build_service(cfg, mock=True, replay=first)

    rows = []
    print(
        f"{'clip':<28}{'truth':<22}{'gate':<7}{'label':<30}{'conf':>6}{'nov':>8}"
        f"{'outcome':>9}{'sec':>6}  verdict"
    )
    print("-" * 132)
    for m in manifest:
        clip = args.clips / m["file"]
        pipeline.recorder.source = clip
        event = MotionEvent.now(trigger=Trigger.MOCK)

        t0 = time.perf_counter()
        rec = pipeline.handle(event)
        secs = time.perf_counter() - t0

        d = rec.decision
        ok, why = is_correct(m["truth"], rec)
        gated = rec.empty
        row = {
            "file": m["file"],
            "truth": m["truth"],
            "gated_empty": gated,
            "recorded": rec.clip_path is not None,
            "label": getattr(d, "label", None),
            "level": getattr(d, "level", None),
            "confidence": round(getattr(d, "confidence", 0.0), 3) if d else None,
            "novelty_score": round(getattr(d, "novelty_score", 0.0), 3) if d else None,
            "outcome": rec.outcome.value if rec.outcome else None,
            "seconds": round(secs, 2),
            "correct": ok,
            "why": why,
            "error": rec.error,
        }
        rows.append(row)
        print(
            f"{m['file'][:27]:<28}{m['truth'][:21]:<22}{'EMPTY' if gated else '-':<7}"
            f"{str(row['label'])[:29]:<30}{row['confidence'] or 0:>6.2f}"
            f"{row['novelty_score'] or 0:>8.2f}{str(row['outcome']):>9}{secs:>6.2f}  "
            f"{'ok ' if ok else 'XX '}{why}"
        )

    empties = [r for r in rows if r["truth"] == "empty_feeder"]
    birds = [r for r in rows if r["truth"] != "empty_feeder"]
    summary = {
        "clips": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "empty_clips_discarded": f"{sum(r['correct'] for r in empties)}/{len(empties)}",
        "bird_clips_correct": f"{sum(r['correct'] for r in birds)}/{len(birds)}",
        "median_seconds_recorded_event": sorted(r["seconds"] for r in rows if r["recorded"])[
            len([r for r in rows if r["recorded"]]) // 2
        ]
        if any(r["recorded"] for r in rows)
        else None,
        "median_seconds_gated_event": sorted(r["seconds"] for r in rows if r["gated_empty"])[
            len([r for r in rows if r["gated_empty"]]) // 2
        ]
        if any(r["gated_empty"] for r in rows)
        else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
