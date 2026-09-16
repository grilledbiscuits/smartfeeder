#!/usr/bin/env python3
"""Reconstruct low-rate test-split clip sequences from existing labelled frames.

These are decoded/re-encoded field frames, not original FHD recordings. Store
source IDs and image hashes so the experiment can be traced to its labels.
Empty candidates are excluded: gaps in their frame numbers need not be visits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ml" / "src")]


def main():
    from birdcam.config import load_config
    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config()
    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m, include_field=True)
    groups = defaultdict(list)
    for item in items:
        if item.source == "field" and item.split == "test" and item.taxon_label != "empty_feeder":
            groups[(item.taxon_label, item.observation_id)].append(item)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for j, ((label, obs), rows) in enumerate(sorted(groups.items())):
        rows = sorted(rows, key=lambda i: i.path.name)[:24]
        filename = f"{j:02d}_{obs}.mp4"
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
                    str(args.out / filename),
                ],
                check=True,
                timeout=60,
            )
        manifest.append(
            dict(
                file=filename,
                truth=label,
                observation_id=obs,
                split="test",
                source="re-encoded labelled field frames, first up to 24 at 2fps",
                frames=len(rows),
                image_ids=[r.image_id for r in rows],
                image_sha256=[hashlib.sha256(r.path.read_bytes()).hexdigest() for r in rows],
            )
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} reconstructed clips to {args.out}")


if __name__ == "__main__":
    main()
