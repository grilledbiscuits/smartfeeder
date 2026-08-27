"""Populate var/feeder.db with realistic dummy visits.

Lets the dashboard be built and tested before the real inference pipeline
writes any records. Also drops placeholder images (Pillow) and, if ffmpeg is
on PATH, short placeholder video clips into var/media/, so the media-serving
routes have real files to show. Safe to re-run: it wipes and recreates
var/feeder.db each time.

Run from the repo root:
    uv run python -m web.scripts.create_dummy_data
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta

from PIL import Image, ImageDraw

from web.db import add_visit, init_db
from web.paths import DB_PATH, IMAGES_DIR, VIDEOS_DIR

# (species, confidence, has_video, duration_seconds)
VISITS = [
    ("Southern Double-collared Sunbird", 0.94, True, 8.6),
    ("Cape Sugarbird", 0.88, True, 12.1),
    ("Malachite Sunbird", 0.91, False, 5.3),
    ("Amethyst Sunbird", 0.76, True, 9.8),
    ("Cape White-eye", 0.82, False, 4.2),
    ("Southern Double-collared Sunbird", 0.97, True, 15.4),
    ("uncertain", 0.41, False, 3.1),
    ("Cape Weaver", 0.85, True, 7.0),
    ("Malachite Sunbird", 0.93, False, 6.6),
    ("Southern Double-collared Sunbird", 0.89, True, 10.2),
]


def _make_placeholder_image(path, label: str) -> None:
    img = Image.new("RGB", (320, 240), color=(76, 112, 76))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), label, fill=(255, 255, 255))
    img.save(path)


def _make_placeholder_video(path) -> bool:
    """Generate a 2s colour+tone clip with ffmpeg. Returns False if ffmpeg
    isn't installed, leaving the caller to record a filename that points at
    nothing -- which exercises the dashboard's missing-file handling."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=steelblue:s=320x240:d=2",
            "-f", "lavfi", "-i", "sine=frequency=800:duration=2",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return True


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    init_db()

    now = datetime.now()
    total = len(VISITS)
    for i, (species, confidence, has_video, duration) in enumerate(VISITS):
        n = i + 1
        timestamp = (now - timedelta(minutes=15 * (total - i))).strftime("%Y-%m-%d %H:%M:%S")

        image_filename = f"visit_{n:04d}.jpg"
        _make_placeholder_image(IMAGES_DIR / image_filename, species)

        video_filename = None
        if has_video:
            video_filename = f"visit_{n:04d}.mp4"
            _make_placeholder_video(VIDEOS_DIR / video_filename)

        add_visit(
            timestamp=timestamp,
            species=species,
            confidence=confidence,
            image_filename=image_filename,
            video_filename=video_filename,
            duration_seconds=duration,
        )

    print(f"Wrote {total} dummy visits to {DB_PATH}")
    print(f"Media in {IMAGES_DIR} and {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
