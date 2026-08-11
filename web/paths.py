"""Filesystem locations shared by the dashboard and its dummy-data script.

Kept in one place so the runtime layout (database + media) can move without
hunting through app.py and the scripts. All paths are relative to the repo
root, computed from this file's location -- nothing hardcoded, nothing
absolute.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Runtime state: the visits database and the media it references. Separate
# from data/, which holds the ML training corpus -- these are unrelated
# concerns that happen to both be SQLite-backed.
VAR_DIR = BASE_DIR / "var"
DB_PATH = VAR_DIR / "feeder.db"

MEDIA_DIR = VAR_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
VIDEOS_DIR = MEDIA_DIR / "videos"
