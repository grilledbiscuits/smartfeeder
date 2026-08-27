"""Durable on-disk state: the pending-publish queue, review clips, and the
storage guards that stop this service filling the SD card.

Three directories, three lifetimes:

* **work/**    -- a recording in progress. Anything here at startup is debris
                  from a crash or a power cut and is deleted: an unfinalised
                  H.264 stream is not a clip.
* **pending/** -- recorded, classified, on the allowlist, publication NOT yet
                  confirmed. **Never deleted to make room.** This is the
                  promise that "never delete a clip whose upload hasn't been
                  confirmed" survives a reboot, so pending is a hard ceiling
                  on recording rather than a cache that can be evicted.
* **review/**  -- clips the classifier abstained on, kept for a human look
                  (see policy.py). These ARE evictable, oldest first, because
                  nothing depends on any individual one.

Every pending clip has a JSON sidecar written BEFORE the first publish attempt
and updated atomically after each one. A clip whose sidecar is missing or
unreadable is not thrown away -- it is re-adopted with a minimal sidecar, since
a clip on disk is evidence and a parse error is not a reason to destroy it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from capture.events import CaptureRecord, SpoolEntry

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".json"


def fsync_file(path: Path) -> None:
    """Flush one file's bytes to the card.

    Without this, a clip can be handed to the classifier -- or worse, reported
    as published -- while its tail is still in the page cache, and a power cut
    at the wrong moment leaves a truncated file that looks complete.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    """Flush a directory entry, so a rename survives a power cut."""
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    if isinstance(obj, Path):
        return obj.name
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return str(obj)


class InsufficientStorage(RuntimeError):
    """Raised when a recording cannot start without risking the card."""


class Spool:
    """Owns the three directories and every deletion decision."""

    def __init__(
        self,
        work_dir: Path,
        pending_dir: Path,
        review_dir: Path,
        *,
        min_free_mb: int,
        max_pending_clips: int,
        max_review_clips: int,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.pending_dir = Path(pending_dir)
        self.review_dir = Path(review_dir)
        self.min_free_mb = int(min_free_mb)
        self.max_pending_clips = int(max_pending_clips)
        self.max_review_clips = int(max_review_clips)
        for d in (self.work_dir, self.pending_dir, self.review_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- storage guards --------------------------------------------------------

    def free_mb(self) -> float:
        return shutil.disk_usage(self.work_dir).free / (1024 * 1024)

    def pending_count(self) -> int:
        return len(list(self.pending_dir.glob("*.mp4")))

    def review_count(self) -> int:
        return len(list(self.review_dir.glob("*.mp4")))

    def check_can_record(self) -> None:
        """Raise InsufficientStorage if starting a recording would be reckless.

        Checked BEFORE the camera is opened rather than after, so a full card
        costs one log line instead of a clip that cannot be written.
        """
        free = self.free_mb()
        if free < self.min_free_mb:
            # Evict review clips first -- they are the only expendable ones --
            # and re-check before giving up.
            self.prune_review(target_count=0)
            free = self.free_mb()
            if free < self.min_free_mb:
                raise InsufficientStorage(
                    f"{free:.0f} MB free, floor is {self.min_free_mb} MB, and no "
                    "review clips left to evict. Pending clips are not deletable: "
                    f"{self.pending_count()} awaiting publication."
                )

        pending = self.pending_count()
        if pending >= self.max_pending_clips:
            raise InsufficientStorage(
                f"{pending} clips awaiting publication, cap is "
                f"{self.max_pending_clips}. Not recording until the backlog "
                "drains -- these clips cannot be deleted, so recording over the "
                "cap would fill the card."
            )

    # -- work directory --------------------------------------------------------

    def work_path(self, event_id: str) -> Path:
        return self.work_dir / f"{event_id}.mp4"

    def clear_work(self) -> int:
        """Delete crash debris. Returns the number of files removed."""
        n = 0
        for p in sorted(self.work_dir.iterdir()):
            if p.is_file():
                logger.warning("discarding unfinalised recording from a previous run: %s", p.name)
                p.unlink(missing_ok=True)
                n += 1
        return n

    # -- pending queue ---------------------------------------------------------

    def _sidecar_for(self, clip: Path) -> Path:
        return clip.with_suffix(SIDECAR_SUFFIX)

    def write_sidecar(self, clip: Path, data: dict[str, Any]) -> Path:
        """Atomically write a sidecar: temp file, fsync, rename, fsync dir."""
        sidecar = self._sidecar_for(clip)
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, default=_json_default), encoding="utf-8")
        fsync_file(tmp)
        os.replace(tmp, sidecar)
        fsync_dir(sidecar.parent)
        return sidecar

    def enqueue(self, record: CaptureRecord) -> SpoolEntry:
        """Move a finished clip into pending/ and write its sidecar.

        Sidecar first, then the clip's move is already done -- ordering here is
        deliberate: a clip in pending/ with no sidecar is recoverable
        (`iter_pending` re-adopts it), a sidecar with no clip is not.
        """
        if record.clip_path is None:
            raise ValueError(f"{record.event_id}: cannot enqueue a record with no clip")

        dest = self.pending_dir / f"{record.event_id}.mp4"
        shutil.move(str(record.clip_path), str(dest))
        record.clip_path = dest

        keyframe_name = None
        if record.keyframe_path and record.keyframe_path.is_file():
            k_dest = self.pending_dir / f"{record.event_id}.jpg"
            shutil.move(str(record.keyframe_path), str(k_dest))
            record.keyframe_path = k_dest
            keyframe_name = k_dest.name

        data = record.log_fields()
        data.update(
            {
                "keyframe": keyframe_name,
                "attempts": 0,
                "last_error": None,
                "visit_id": None,
                "enqueued_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        sidecar = self.write_sidecar(dest, data)
        fsync_dir(self.pending_dir)
        return SpoolEntry(clip_path=dest, sidecar_path=sidecar, data=data)

    def iter_pending(self) -> list[SpoolEntry]:
        """Every unconfirmed clip, oldest first.

        Called at startup as well as after each publish, so a service that was
        killed mid-backoff resumes its queue instead of stranding it.
        """
        entries: list[SpoolEntry] = []
        for clip in sorted(self.pending_dir.glob("*.mp4")):
            sidecar = self._sidecar_for(clip)
            data: dict[str, Any]
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                # Re-adopt rather than delete. The clip is real footage; a
                # missing sidecar only costs us its metadata.
                logger.warning(
                    "%s has no readable sidecar (%s); re-adopting with minimal metadata",
                    clip.name,
                    type(exc).__name__,
                )
                data = {
                    "event_id": clip.stem,
                    "label": None,
                    "confidence": None,
                    "attempts": 0,
                    "recovered": True,
                }
                self.write_sidecar(clip, data)
            entries.append(SpoolEntry(clip_path=clip, sidecar_path=sidecar, data=data))
        entries.sort(key=lambda e: str(e.data.get("enqueued_at", e.clip_path.stem)))
        return entries

    def record_attempt(self, entry: SpoolEntry, error: str | None) -> None:
        entry.data["attempts"] = entry.attempts + 1
        entry.data["last_error"] = error
        entry.data["last_attempt_at"] = datetime.now().isoformat(timespec="seconds")
        self.write_sidecar(entry.clip_path, entry.data)

    def mark_published(self, entry: SpoolEntry, visit_id: int) -> None:
        """Record the visit id BEFORE any deletion.

        This is the confirmation that makes deletion safe, so it is persisted
        first. A crash between here and `release` leaves a published clip in
        pending/, which the next pass recognises by its visit_id and cleans up
        -- the harmless direction.
        """
        entry.data["visit_id"] = int(visit_id)
        entry.data["published_at"] = datetime.now().isoformat(timespec="seconds")
        self.write_sidecar(entry.clip_path, entry.data)

    def release(self, entry: SpoolEntry, *, delete: bool) -> None:
        """Drop a confirmed-published clip from the queue."""
        if entry.visit_id is None:
            raise ValueError(
                f"{entry.event_id}: refusing to release a clip with no visit_id. "
                "Publication is unconfirmed."
            )
        keyframe = entry.keyframe_path
        if delete:
            entry.clip_path.unlink(missing_ok=True)
            if keyframe:
                keyframe.unlink(missing_ok=True)
            entry.sidecar_path.unlink(missing_ok=True)
        else:
            # Retained locally: move out of pending/ so it stops counting
            # against the recording cap, but keep the sidecar as provenance.
            self.review_dir.mkdir(parents=True, exist_ok=True)
            for src in (entry.clip_path, keyframe, entry.sidecar_path):
                if src and src.is_file():
                    shutil.move(str(src), str(self.review_dir / src.name))
            self.prune_review()

    # -- review directory ------------------------------------------------------

    def retain_for_review(self, record: CaptureRecord) -> Path | None:
        """Keep an abstained clip for a human look, under its own cap."""
        if record.clip_path is None or not record.clip_path.is_file():
            return None
        dest = self.review_dir / f"{record.event_id}.mp4"
        shutil.move(str(record.clip_path), str(dest))
        record.clip_path = dest
        if record.keyframe_path and record.keyframe_path.is_file():
            k_dest = self.review_dir / f"{record.event_id}.jpg"
            shutil.move(str(record.keyframe_path), str(k_dest))
            record.keyframe_path = k_dest
        self.write_sidecar(dest, record.log_fields())
        self.prune_review()
        return dest

    def prune_review(self, target_count: int | None = None) -> int:
        """Evict oldest review clips down to the cap. Returns how many went."""
        cap = self.max_review_clips if target_count is None else target_count
        clips = sorted(self.review_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        excess = len(clips) - cap
        removed = 0
        for clip in clips[: max(0, excess)]:
            clip.unlink(missing_ok=True)
            clip.with_suffix(SIDECAR_SUFFIX).unlink(missing_ok=True)
            clip.with_suffix(".jpg").unlink(missing_ok=True)
            removed += 1
        if removed:
            logger.info("pruned %d review clip(s) to stay within cap %d", removed, cap)
        return removed

    # -- discard ---------------------------------------------------------------

    def discard(self, record: CaptureRecord) -> None:
        """Delete a clip that is not of interest. Immediate, per the brief."""
        for p in (record.clip_path, record.keyframe_path):
            if p and p.is_file():
                p.unlink(missing_ok=True)
