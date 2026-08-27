"""Publishing a visit to the dashboard.

The interface contract is the one `web/db.py` documents in its own docstring:
"the future feeder/inference process is expected to call add_visit() once it
has classified a visit and saved its media -- that's the whole interface
contract between the two sides." There is no upload endpoint, no auth and no
multipart in `web/`, and this package does not add any: it writes the media
into `var/media/` and inserts one row, which is what the dashboard reads.

SQLite is already in WAL mode (`web/db.py`) specifically so this process can
write while the dashboard reads, so no coordination beyond that is needed.

Ordering, and the one duplicate-row window
------------------------------------------
Media first, then the row. A row whose media is missing renders as "no video"
-- the dashboard checks `is_file()` -- whereas media with no row is invisible
and leaks disk forever. So the recoverable direction is chosen deliberately.

The clip is COPIED out of the spool rather than moved: the spool copy is the
one that may not be deleted until publication is confirmed, and moving it would
break that promise for the duration of the insert.

There is one window left: a crash between a successful INSERT and the sidecar
recording its `visit_id` re-publishes that clip on restart, producing a
duplicate row. It is one row, it is visible, and closing it properly would mean
a unique constraint on `visits` -- a schema change to the other side of a fixed
interface. Noted in the README instead.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from capture.events import SpoolEntry
from capture.labels import display_name
from capture.spool import fsync_dir, fsync_file

logger = logging.getLogger(__name__)

# The existing rows use this format (web/scripts/create_dummy_data.py) and the
# dashboard orders by the raw TEXT column, so an ISO 'T' separator here would
# sort every new row against the old ones incorrectly.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class PublishError(RuntimeError):
    """Publication failed. The clip stays in the spool and is retried."""


class VisitPublisher(Protocol):
    def publish(self, entry: SpoolEntry) -> int: ...


class LocalDashboardPublisher:
    """Same-host publication: media into var/media/, one row via add_visit()."""

    def __init__(
        self,
        images_dir: Path,
        videos_dir: Path,
        add_visit,
        *,
        birdcam_config: Any | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.videos_dir = Path(videos_dir)
        self.add_visit = add_visit
        self.birdcam_config = birdcam_config
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def _place(self, src: Path, dest_dir: Path, name: str) -> str:
        """Copy one media file into the dashboard's tree and flush it."""
        dest = dest_dir / name
        shutil.copyfile(src, dest)
        fsync_file(dest)
        fsync_dir(dest_dir)
        return dest.name

    def publish(self, entry: SpoolEntry) -> int:
        if entry.visit_id is not None:
            # A previous attempt inserted the row and crashed before the spool
            # could be released. Do not insert a second one.
            logger.info(
                "%s already has visit_id=%d; skipping insert", entry.event_id, entry.visit_id
            )
            return entry.visit_id

        if not entry.clip_path.is_file():
            raise PublishError(f"{entry.event_id}: clip is gone from the spool")

        data = entry.data
        try:
            video_name = self._place(entry.clip_path, self.videos_dir, f"{entry.event_id}.mp4")

            image_name = None
            keyframe = entry.keyframe_path
            if keyframe and keyframe.is_file():
                image_name = self._place(keyframe, self.images_dir, f"{entry.event_id}.jpg")
        except OSError as exc:
            raise PublishError(
                f"{entry.event_id}: could not write media into {self.videos_dir.parent}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        label = data.get("label") or "uncertain"
        species = display_name(label, self.birdcam_config)
        timestamp = _format_timestamp(data.get("detected_at"))

        try:
            visit_id = self.add_visit(
                timestamp=timestamp,
                species=species,
                confidence=float(data.get("confidence") or 0.0),
                image_filename=image_name,
                video_filename=video_name,
                duration_seconds=data.get("duration_s"),
            )
        except Exception as exc:
            raise PublishError(
                f"{entry.event_id}: add_visit failed: {type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "published visit %d: %s (%s) conf=%.3f",
            visit_id,
            species,
            label,
            float(data.get("confidence") or 0.0),
        )
        return int(visit_id)


def _format_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime(TIMESTAMP_FORMAT)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value).strftime(TIMESTAMP_FORMAT)
        except ValueError:
            return value
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def build_local_publisher(birdcam_config: Any | None = None) -> LocalDashboardPublisher:
    """Wire the publisher to the real `web` package.

    Imported lazily so the rest of this package -- and its tests -- never
    require Flask or the dashboard's runtime state to be present.
    """
    try:
        from web.db import add_visit, init_db
        from web.paths import IMAGES_DIR, VIDEOS_DIR
    except ImportError as exc:
        raise PublishError(
            "could not import the `web` package. Run the service from the repo "
            "root, or install it, so `web.db` and `web.paths` resolve."
        ) from exc

    init_db()  # CREATE TABLE IF NOT EXISTS; safe on every start
    return LocalDashboardPublisher(
        images_dir=IMAGES_DIR,
        videos_dir=VIDEOS_DIR,
        add_visit=add_visit,
        birdcam_config=birdcam_config,
    )
