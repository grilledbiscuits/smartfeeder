#!/usr/bin/env python3
"""Reconstruct low-rate test-split clip sequences from existing labelled frames.

These are decoded/re-encoded field frames, not original FHD recordings. Store
source IDs and image hashes so the experiment can be traced to its labels.

Pillarboxed frames are dropped (2026-09-21): portrait video padded into a
landscape frame is a geometry the Pi's camera cannot produce, so a replay built
from them measures an input the deployment never sees. In the 2026-09-16 soak,
two of the five wrong actions were pillarboxed Southern Double-collared clips
called `other_animal`. Detection is per frame -- some sessions hold both kinds.
`--include-letterboxed` keeps them, for comparing against that soak.

Empty clips (`--empty-runs`)
----------------------------
Empty frames are only assembled from CONSECUTIVE frame numbers: a gap between
two empty frames may hide a visit, so a clip stitched across one would not be
an empty clip. They are written per session with a `reference.jpg` from the same
session, because the gate compares a clip against a background of its own scene
-- an empty clip replayed against another session's background reads as
occupied, and measures nothing. Run each session's directory as its own replay.

The reference frame is a member of that session's empty frames, so it is a seed,
never evidence: exclude its own clip from any reported score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ml" / "src")]


def frame_number(item) -> int:
    match = re.search(r"(\d+)\.jpg$", item.path.name)
    return int(match.group(1)) if match else -1


def consecutive_runs(rows, min_length: int) -> list[list]:
    """Split frames into runs of consecutive frame numbers."""
    runs, current = [], []
    for row in sorted(rows, key=frame_number):
        if current and frame_number(row) != frame_number(current[-1]) + 1:
            runs.append(current)
            current = []
        current.append(row)
    runs.append(current)
    return [r for r in runs if len(r) >= min_length]


def encode(rows, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for n, row in enumerate(rows):
            (Path(tmp) / f"f_{n:05d}.jpg").symlink_to(row.path.resolve())
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "2",
                "-framerate",
                "2",
                "-i",
                str(Path(tmp) / "f_%05d.jpg"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ],  # fmt: skip
            check=True,
            timeout=60,
        )


def entry(rows, filename: str, truth: str, obs: str, session: str) -> dict:
    return dict(
        file=filename,
        truth=truth,
        observation_id=obs,
        session=session,
        split="test",
        source="re-encoded labelled field frames at 2fps",
        letterboxed=False,
        frames=len(rows),
        image_ids=[r.image_id for r in rows],
        image_sha256=[hashlib.sha256(r.path.read_bytes()).hexdigest() for r in rows],
    )


def main():
    from birdcam.config import load_config
    from birdcam.data.dataset import load_labelled
    from birdcam.data.letterbox import letterboxed_ids, session_of
    from birdcam.data.manifest import open_manifest

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-letterboxed", action="store_true")
    ap.add_argument("--min-frames", type=int, default=4, help="skip visits shorter than this")
    ap.add_argument(
        "--empty-runs",
        type=int,
        default=0,
        metavar="N",
        help="also build empty clips from runs of at least N consecutive empty frames",
    )
    args = ap.parse_args()

    cfg = load_config()
    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m, include_field=True)
    barred = (
        set()
        if args.include_letterboxed
        else letterboxed_ids(items, cache_path=ROOT / "ml/data/letterboxed.json")
    )

    test = [
        i for i in items if i.source == "field" and i.split == "test" and i.image_id not in barred
    ]
    dropped = sum(
        1 for i in items if i.source == "field" and i.split == "test" and i.image_id in barred
    )

    groups = defaultdict(list)
    for item in test:
        if item.taxon_label != "empty_feeder":
            groups[(item.taxon_label, item.observation_id)].append(item)
    thin = [k for k, v in groups.items() if len(v) < args.min_frames]
    for k in thin:
        del groups[k]

    args.out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for j, ((label, obs), rows) in enumerate(sorted(groups.items())):
        rows = sorted(rows, key=lambda i: i.path.name)[:24]
        filename = f"{j:02d}_{obs}.mp4"
        encode(rows, args.out / filename)
        manifest.append(entry(rows, filename, label, obs, session_of(rows[0])))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote {len(manifest)} visit clips to {args.out} "
        f"({dropped} pillarboxed frames dropped, {len(thin)} visits left too short)"
    )

    if not args.empty_runs:
        return

    # Empty clips, per session, each with a background reference of its own scene.
    by_session = defaultdict(list)
    for item in test:
        if item.taxon_label == "empty_feeder":
            by_session[session_of(item)].append(item)

    for session, rows in sorted(by_session.items()):
        runs = consecutive_runs(rows, args.empty_runs)
        if len(runs) < 2:
            print(
                f"  {session}: {len(runs)} usable empty run(s) -- need one to seed the "
                "background and at least one to score; skipped"
            )
            continue
        out = args.out.parent / f"gate-{session}"
        out.mkdir(parents=True, exist_ok=False)
        seed, scored = runs[0], runs[1:]
        entries = []
        for k, run in enumerate(scored):
            filename = f"empty_{k:02d}.mp4"
            encode(run[:24], out / filename)
            entries.append(entry(run[:24], filename, "empty_feeder", session, session))
        for row in manifest:
            if row["session"] == session:
                (out / row["file"]).symlink_to((args.out / row["file"]).resolve())
                entries.append(row)
        (out / "manifest.json").write_text(json.dumps(entries, indent=2))

        from PIL import Image

        with Image.open(seed[0].path) as im:
            im.convert("RGB").save(out / "reference.jpg", quality=95)
        (out / "reference.json").write_text(
            json.dumps(
                dict(
                    image_id=seed[0].image_id,
                    session=session,
                    note="seed for empty_gate.reference; excluded from the scored clips",
                ),
                indent=2,
            )
        )
        print(
            f"  {out}: {len(scored)} empty clip(s) + "
            f"{sum(1 for e in entries if e['truth'] != 'empty_feeder')} visit clip(s), "
            f"seeded from {seed[0].path.name}"
        )


if __name__ == "__main__":
    main()
