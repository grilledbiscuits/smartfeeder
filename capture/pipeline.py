"""The per-event flow: record -> classify -> decide -> publish / retain / discard.

This is the hardware-free core. Everything it touches is injected -- recorder,
classifier, publisher, spool, clock -- so the whole decision path can be driven
off-Pi with a replayed clip and a stub model.

No stage is allowed to raise out of `handle`. A camera failure, a missing
model, a full card and a locked database are all ordinary conditions for a
service that runs unattended for weeks; each is logged with the event id and
leaves the pipeline ready for the next trigger.

Retry policy
------------
`escalate_after_attempts` is not a give-up count. A clip whose publication is
unconfirmed is never abandoned and never deleted, so retries continue
indefinitely at `backoff_max_seconds`; the count only controls when the failure
stops being an INFO-level fact and starts being an ERROR the operator should
see. The thing that bounds disk use is `storage.max_pending_clips`, which stops
new recordings rather than discarding old evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from capture.classifier import ClassifierUnavailable
from capture.events import CaptureRecord, MotionEvent, Outcome, SpoolEntry
from capture.logging_setup import log_event
from capture.policy import decide_outcome, describe
from capture.publisher import PublishError
from capture.recorder import CameraUnavailable
from capture.spool import InsufficientStorage, Spool

logger = logging.getLogger(__name__)


class CapturePipeline:
    """One motion event in, one CaptureRecord out."""

    def __init__(
        self,
        spool: Spool,
        recorder: Any,
        classifier: Any | None,
        publisher: Any,
        *,
        clip_seconds: float,
        retain_uncertain: bool,
        delete_after_publish: bool,
        escalate_after_attempts: int,
        backoff_initial_seconds: float,
        backoff_max_seconds: float,
        backoff_factor: float,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.spool = spool
        self.recorder = recorder
        self.classifier = classifier
        self.publisher = publisher
        self.clip_seconds = float(clip_seconds)
        self.retain_uncertain = bool(retain_uncertain)
        self.delete_after_publish = bool(delete_after_publish)
        self.escalate_after_attempts = int(escalate_after_attempts)
        self.backoff_initial = float(backoff_initial_seconds)
        self.backoff_max = float(backoff_max_seconds)
        self.backoff_factor = float(backoff_factor)
        self.now = now

    # -- the event flow --------------------------------------------------------

    def handle(self, event: MotionEvent) -> CaptureRecord:
        record = CaptureRecord(event=event)

        try:
            self.spool.check_can_record()
        except InsufficientStorage as exc:
            record.error = f"storage: {exc}"
            logger.error("%s: not recording -- %s", event.event_id, exc)
            log_event(logger, logging.ERROR, "capture skipped", record.log_fields())
            return record

        if not self._record_clip(record):
            log_event(logger, logging.ERROR, "capture failed", record.log_fields())
            return record

        self._classify(record)

        record.outcome = decide_outcome(record.decision, retain_uncertain=self.retain_uncertain)
        logger.info("%s: %s", event.event_id, describe(record.decision, record.outcome))

        try:
            self._act(record)
        except OSError as exc:
            # A failure to move or delete a file is a storage problem, not a
            # reason to lose the service.
            record.error = f"{type(exc).__name__}: {exc}"
            logger.exception("%s: could not apply outcome %s", event.event_id, record.outcome)

        log_event(logger, logging.INFO, "capture decided", record.log_fields())
        return record

    def _record_clip(self, record: CaptureRecord) -> bool:
        dest = self.spool.work_path(record.event_id)
        try:
            result = self.recorder.record(dest, self.clip_seconds)
        except CameraUnavailable as exc:
            record.error = f"camera: {exc}"
            logger.error("%s: recording failed -- %s", record.event_id, exc)
            dest.unlink(missing_ok=True)
            return False
        except Exception as exc:  # noqa: BLE001 - the service must survive anything here
            record.error = f"{type(exc).__name__}: {exc}"
            logger.exception("%s: unexpected recording failure", record.event_id)
            dest.unlink(missing_ok=True)
            return False

        record.clip_path = result.path
        record.duration_seconds = result.duration_seconds
        logger.info(
            "%s: recorded %.1fs, %.1f MB",
            record.event_id,
            result.duration_seconds,
            result.size_bytes / 1e6,
        )
        return True

    def _classify(self, record: CaptureRecord) -> None:
        if self.classifier is None:
            # Classification disabled: no verdict, so policy retains the clip
            # rather than guessing. Useful for collecting footage before the
            # model is deployable.
            record.error = "classifier disabled"
            return
        try:
            result = self.classifier.classify(
                record.clip_path, self.spool.work_dir, record.event_id
            )
        except ClassifierUnavailable as exc:
            record.error = f"classifier: {exc}"
            logger.error("%s: classification failed -- %s", record.event_id, exc)
            return
        except Exception as exc:  # noqa: BLE001 - never lose the service to a bad frame
            record.error = f"{type(exc).__name__}: {exc}"
            logger.exception("%s: unexpected classification failure", record.event_id)
            return

        record.decision = result.decision
        record.frames_scored = result.frames_scored
        record.keyframe_path = result.keyframe

    def _act(self, record: CaptureRecord) -> None:
        if record.outcome is Outcome.DISCARD:
            self.spool.discard(record)
            return

        if record.outcome is Outcome.RETAIN:
            kept = self.spool.retain_for_review(record)
            if kept:
                logger.info("%s: retained for review at %s", record.event_id, kept)
            return

        entry = self.spool.enqueue(record)
        self._attempt(entry)

    # -- the pending queue -----------------------------------------------------

    def drain_pending(self) -> int:
        """Try every clip whose backoff has elapsed. Returns how many published.

        Called on startup (so a queue that outlived a reboot resumes) and after
        each event.
        """
        published = 0
        for entry in self.spool.iter_pending():
            if not self._due(entry):
                continue
            if self._attempt(entry):
                published += 1
        return published

    def _due(self, entry: SpoolEntry) -> bool:
        due_at = entry.data.get("next_attempt_at")
        if not due_at:
            return True
        try:
            return self.now() >= datetime.fromisoformat(str(due_at))
        except ValueError:
            return True

    def _attempt(self, entry: SpoolEntry) -> bool:
        """One publish attempt. Never raises; schedules the next try on failure."""
        try:
            visit_id = self.publisher.publish(entry)
        except PublishError as exc:
            self._schedule_retry(entry, str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - a publisher bug must not lose the clip
            self._schedule_retry(entry, f"{type(exc).__name__}: {exc}")
            return False

        # Confirmation is persisted BEFORE anything is deleted.
        self.spool.mark_published(entry, visit_id)
        self.spool.release(entry, delete=self.delete_after_publish)
        log_event(
            logger,
            logging.INFO,
            "publish confirmed",
            {
                "event_id": entry.event_id,
                "visit_id": visit_id,
                "label": entry.data.get("label"),
                "local_copy": "deleted" if self.delete_after_publish else "retained",
            },
        )
        return True

    def _schedule_retry(self, entry: SpoolEntry, error: str) -> None:
        self.spool.record_attempt(entry, error)
        attempts = entry.attempts
        delay = min(
            self.backoff_initial * (self.backoff_factor ** max(0, attempts - 1)),
            self.backoff_max,
        )
        entry.data["next_attempt_at"] = (self.now() + timedelta(seconds=delay)).isoformat(
            timespec="seconds"
        )
        self.spool.write_sidecar(entry.clip_path, entry.data)

        level = logging.ERROR if attempts >= self.escalate_after_attempts else logging.WARNING
        log_event(
            logger,
            level,
            "publish failed, clip kept and requeued",
            {
                "event_id": entry.event_id,
                "attempts": attempts,
                "retry_in_s": round(delay, 1),
                "error": error,
            },
        )
