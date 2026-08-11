"""Flask dashboard: reads visit records from SQLite and serves their media.

This process only reads. It never runs the classifier, never imports
anything from the birdcam package, and has no state of its own beyond what's
in var/feeder.db -- so it can be stopped, restarted, or left crashed without
affecting the feeder.
"""

from __future__ import annotations

from flask import Flask, abort, render_template, send_from_directory

from web.db import get_recent_visits
from web.paths import IMAGES_DIR, VIDEOS_DIR

app = Flask(__name__)


@app.route("/")
def index():
    visits = []
    for row in get_recent_visits(limit=20):
        visit = dict(row)
        visit["confidence_pct"] = round(visit["confidence"] * 100, 1)
        visit["image_exists"] = bool(visit["image_filename"]) and (
            IMAGES_DIR / visit["image_filename"]
        ).is_file()
        visit["video_exists"] = bool(visit["video_filename"]) and (
            VIDEOS_DIR / visit["video_filename"]
        ).is_file()
        visits.append(visit)
    return render_template("index.html", visits=visits)


@app.route("/media/images/<path:filename>")
def image(filename):
    # send_from_directory resolves filename against IMAGES_DIR and rejects
    # any path that escapes it (e.g. "../..."), so this is safe against
    # arbitrary filesystem access.
    if not (IMAGES_DIR / filename).is_file():
        abort(404)
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/media/videos/<path:filename>")
def video(filename):
    if not (VIDEOS_DIR / filename).is_file():
        abort(404)
    return send_from_directory(VIDEOS_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
