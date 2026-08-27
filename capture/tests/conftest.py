"""Shared fixtures. No hardware, no model, no dashboard.

`make_decision` builds a real `birdcam.inference.Decision` rather than a stand-in
so the tests fail if that contract changes -- which is the whole point of
testing against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "ml" / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from birdcam.inference import Decision  # noqa: E402
from capture.events import MotionEvent, Trigger  # noqa: E402
from capture.recorder import CameraUnavailable, RecordingResult  # noqa: E402
from capture.spool import Spool  # noqa: E402


@pytest.fixture
def make_decision():
    def _make(
        label: str = "cinnyris_chalybeus",
        *,
        level: str = "species",
        confidence: float = 0.9,
        is_unknown: bool = False,
        is_capture_target: bool = True,
        novelty_score: float = 0.0,
        sex_label: str | None = "male_breeding",
    ) -> Decision:
        return Decision(
            label=label,
            level=level,
            confidence=confidence,
            sex_label=sex_label,
            sex_confidence=0.7,
            novelty_score=novelty_score,
            is_unknown=is_unknown,
            is_capture_target=is_capture_target,
            top_k=[(label, confidence)],
        )

    return _make


@pytest.fixture
def event():
    return MotionEvent.now(Trigger.MOCK)


@pytest.fixture
def spool(tmp_path) -> Spool:
    return Spool(
        work_dir=tmp_path / "work",
        pending_dir=tmp_path / "pending",
        review_dir=tmp_path / "review",
        min_free_mb=1,
        max_pending_clips=5,
        max_review_clips=3,
    )


class FakeRecorder:
    """Writes a plausible file instead of driving a camera."""

    def __init__(self, *, fail: bool = False, size: int = 4096) -> None:
        self.fail = fail
        self.size = size
        self.calls = 0
        self.closed = False

    def record(self, dest: Path, seconds: float) -> RecordingResult:
        self.calls += 1
        if self.fail:
            raise CameraUnavailable("camera is busy (simulated)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * self.size)
        return RecordingResult(path=dest, duration_seconds=seconds, size_bytes=self.size)

    def close(self) -> None:
        self.closed = True


class FakeClassifier:
    """Returns a preset decision, or raises."""

    def __init__(self, decision=None, *, error: Exception | None = None, keyframe: bool = False):
        self.decision = decision
        self.error = error
        self.keyframe = keyframe
        self.calls = 0

    def classify(self, clip: Path, work_dir: Path, event_id: str):
        from capture.classifier import ClipResult

        self.calls += 1
        if self.error is not None:
            raise self.error
        keyframe = None
        if self.keyframe:
            keyframe = work_dir / f"{event_id}.jpg"
            keyframe.write_bytes(b"\xff\xd8\xff")  # a JPEG magic number is enough
        return ClipResult(decision=self.decision, frames_scored=4, keyframe=keyframe)


class FakePublisher:
    """Records what it was asked to publish; can be told to fail N times."""

    def __init__(self, *, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.published: list[str] = []
        self.attempts = 0

    def publish(self, entry) -> int:
        from capture.publisher import PublishError

        self.attempts += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise PublishError("dashboard unreachable (simulated)")
        self.published.append(entry.event_id)
        return len(self.published)


@pytest.fixture
def fake_recorder():
    return FakeRecorder()


@pytest.fixture
def fake_publisher():
    return FakePublisher()
