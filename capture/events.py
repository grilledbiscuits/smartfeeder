"""The objects that move through the pipeline.

One motion event becomes one CaptureRecord, which accumulates state as it
passes through recording, classification and the keep/discard decision. It is
deliberately a plain mutable dataclass rather than a chain of immutable
values: when something fails halfway, the partially-filled record is what the
log needs to describe what happened.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Outcome(StrEnum):
    """What the pipeline decided to do with a clip.

    RETAIN exists because `uncertain` is not the same as "not interesting".
    See birdcam/inference.py: `uncertain` means "probably a bird, but I cannot
    pin it down", and those are the visits worth a human look. Discarding them
    throws away the only data that would fix the under-triggering measured in
    ASSUMPTIONS.md A27.
    """

    PUBLISH = "publish"  # on the capture allowlist -> dashboard
    RETAIN = "retain"  # kept locally for review, not published
    DISCARD = "discard"  # deleted immediately


class Trigger(StrEnum):
    """Why an event entered the pipeline."""

    PIR = "pir"
    MOCK = "mock"
    MANUAL = "manual"  # one-shot --classify of an existing clip


@dataclass(frozen=True)
class MotionEvent:
    """One admitted motion trigger.

    `event_id` is generated at admission, not at recording, so a queued event
    can be correlated across the drop/record/classify logs.
    """

    event_id: str
    detected_at: datetime
    trigger: Trigger = Trigger.PIR

    @classmethod
    def now(cls, trigger: Trigger = Trigger.PIR, clock=datetime.now) -> MotionEvent:
        ts = clock()
        # Timestamp prefix keeps the id sortable and human-readable in a log;
        # the uuid suffix makes it collision-proof when two events land inside
        # the same second, which the queue permits.
        return cls(
            event_id=f"{ts:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}",
            detected_at=ts,
            trigger=trigger,
        )


@dataclass
class CaptureRecord:
    """One event's journey, filled in stage by stage."""

    event: MotionEvent
    clip_path: Path | None = None
    keyframe_path: Path | None = None
    duration_seconds: float | None = None
    frames_scored: int = 0
    outcome: Outcome | None = None
    error: str | None = None
    # birdcam.inference.Decision, kept untyped so this module does not import
    # numpy just to describe a field.
    decision: Any | None = None

    @property
    def event_id(self) -> str:
        return self.event.event_id

    def log_fields(self) -> dict[str, Any]:
        """Flat dict for the structured log. Never raises on a partial record."""
        d = self.decision
        return {
            "event_id": self.event_id,
            "trigger": self.event.trigger.value,
            "detected_at": self.event.detected_at.isoformat(timespec="seconds"),
            "clip": self.clip_path.name if self.clip_path else None,
            "duration_s": self.duration_seconds,
            "frames_scored": self.frames_scored,
            "label": getattr(d, "label", None),
            "taxon_level": getattr(d, "level", None),
            "confidence": round(getattr(d, "confidence", 0.0), 4) if d else None,
            "sex_label": getattr(d, "sex_label", None),
            "novelty_score": round(getattr(d, "novelty_score", 0.0), 4) if d else None,
            "is_unknown": getattr(d, "is_unknown", None),
            "is_capture_target": getattr(d, "is_capture_target", None),
            "outcome": self.outcome.value if self.outcome else None,
            "error": self.error,
        }


@dataclass
class SpoolEntry:
    """A clip on disk awaiting publication, plus its durable sidecar state.

    The sidecar is what makes "never delete a clip whose upload hasn't been
    confirmed" survive a power cut: the clip and its metadata are both on disk
    before the first publish attempt, so a restart re-queues it.
    """

    clip_path: Path
    sidecar_path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        return str(self.data.get("event_id", self.clip_path.stem))

    @property
    def attempts(self) -> int:
        return int(self.data.get("attempts", 0))

    @property
    def visit_id(self) -> int | None:
        v = self.data.get("visit_id")
        return int(v) if v is not None else None

    @property
    def keyframe_path(self) -> Path | None:
        k = self.data.get("keyframe")
        return self.clip_path.parent / k if k else None
