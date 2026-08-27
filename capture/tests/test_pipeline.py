"""The full per-event flow, with every dependency faked.

Between them these cover the requirement that the service survives camera-busy,
classifier failure, disk-full and network-down without losing a clip or the
process.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from capture.events import Outcome
from capture.pipeline import CapturePipeline
from capture.tests.conftest import FakeClassifier, FakePublisher, FakeRecorder


def build(spool, *, decision=None, recorder=None, classifier=None, publisher=None, **kwargs):
    params = {
        "clip_seconds": 8.0,
        "retain_uncertain": True,
        "delete_after_publish": True,
        "escalate_after_attempts": 3,
        "backoff_initial_seconds": 5.0,
        "backoff_max_seconds": 900.0,
        "backoff_factor": 2.0,
    }
    params.update(kwargs)
    return CapturePipeline(
        spool=spool,
        recorder=recorder or FakeRecorder(),
        classifier=classifier if classifier is not None else FakeClassifier(decision),
        publisher=publisher or FakePublisher(),
        **params,
    )


def test_allowlisted_clip_is_published_and_local_copy_deleted(spool, event, make_decision):
    publisher = FakePublisher()
    pipe = build(spool, decision=make_decision("cinnyris_chalybeus"), publisher=publisher)

    record = pipe.handle(event)

    assert record.outcome is Outcome.PUBLISH
    assert publisher.published == [event.event_id]
    assert spool.pending_count() == 0  # released after confirmation
    assert not any(spool.work_dir.iterdir())


def test_publish_can_keep_the_local_copy(spool, event, make_decision):
    pipe = build(spool, decision=make_decision(), delete_after_publish=False)
    pipe.handle(event)
    assert spool.pending_count() == 0
    assert (spool.review_dir / f"{event.event_id}.mp4").is_file()


def test_uninteresting_clip_is_deleted_immediately(spool, event, make_decision):
    publisher = FakePublisher()
    pipe = build(
        spool,
        decision=make_decision("pycnonotus_capensis", is_capture_target=False),
        publisher=publisher,
    )

    record = pipe.handle(event)

    assert record.outcome is Outcome.DISCARD
    assert publisher.published == []
    assert spool.pending_count() == 0
    assert spool.review_count() == 0
    assert not any(spool.work_dir.iterdir())


def test_uncertain_clip_is_retained_for_review(spool, event, make_decision):
    pipe = build(
        spool,
        decision=make_decision("uncertain", level="uncertain", is_capture_target=False),
    )
    record = pipe.handle(event)

    assert record.outcome is Outcome.RETAIN
    assert (spool.review_dir / f"{event.event_id}.mp4").is_file()
    assert spool.pending_count() == 0


def test_camera_failure_does_not_raise_and_leaves_no_debris(spool, event):
    pipe = build(spool, recorder=FakeRecorder(fail=True))
    record = pipe.handle(event)

    assert record.error is not None
    assert "camera" in record.error
    assert record.outcome is None
    assert not any(spool.work_dir.iterdir())


def test_classifier_failure_retains_rather_than_deletes(spool, event):
    pipe = build(spool, classifier=FakeClassifier(error=RuntimeError("onnx exploded")))
    record = pipe.handle(event)

    assert record.decision is None
    assert record.outcome is Outcome.RETAIN
    assert (spool.review_dir / f"{event.event_id}.mp4").is_file()


def test_full_disk_skips_recording_entirely(spool, event, monkeypatch, make_decision):
    monkeypatch.setattr(spool, "free_mb", lambda: 0.0)
    recorder = FakeRecorder()
    pipe = build(spool, decision=make_decision(), recorder=recorder)

    record = pipe.handle(event)

    assert recorder.calls == 0  # the camera is never even opened
    assert record.error is not None and record.error.startswith("storage:")


def test_publish_failure_keeps_the_clip_queued(spool, event, make_decision):
    publisher = FakePublisher(fail_times=1)
    pipe = build(spool, decision=make_decision(), publisher=publisher)

    pipe.handle(event)

    assert spool.pending_count() == 1  # NOT deleted
    entry = spool.iter_pending()[0]
    assert entry.attempts == 1
    assert entry.visit_id is None
    assert "next_attempt_at" in entry.data
    assert "unreachable" in entry.data["last_error"]


def test_backoff_defers_the_next_attempt(spool, event, make_decision):
    now = datetime(2026, 8, 27, 12, 0, 0)
    publisher = FakePublisher(fail_times=5)
    pipe = build(spool, decision=make_decision(), publisher=publisher, now=lambda: now)

    pipe.handle(event)
    attempts_after_first = publisher.attempts

    # Still inside the backoff window: drain must not retry.
    assert pipe.drain_pending() == 0
    assert publisher.attempts == attempts_after_first


def test_retry_succeeds_once_the_backoff_elapses(spool, event, make_decision):
    clock = {"now": datetime(2026, 8, 27, 12, 0, 0)}
    publisher = FakePublisher(fail_times=1)
    pipe = build(
        spool, decision=make_decision(), publisher=publisher, now=lambda: clock["now"]
    )

    pipe.handle(event)
    assert spool.pending_count() == 1

    clock["now"] += timedelta(seconds=60)
    assert pipe.drain_pending() == 1
    assert publisher.published == [event.event_id]
    assert spool.pending_count() == 0


def test_a_queue_that_outlived_a_restart_is_drained(spool, event, make_decision):
    """Startup recovery: pending clips from a previous process publish."""
    failing = build(spool, decision=make_decision(), publisher=FakePublisher(fail_times=1))
    failing.handle(event)
    assert spool.pending_count() == 1

    # A fresh pipeline, as after a reboot. No next_attempt_at is honoured
    # because the sidecar's window has long passed in real time.
    entry = spool.iter_pending()[0]
    entry.data["next_attempt_at"] = "2020-01-01T00:00:00"
    spool.write_sidecar(entry.clip_path, entry.data)

    publisher = FakePublisher()
    restarted = build(spool, publisher=publisher)
    assert restarted.drain_pending() == 1
    assert publisher.published == [event.event_id]


def test_published_clip_is_not_inserted_twice_after_a_crash(spool, event, make_decision):
    """The sidecar's visit_id is what makes a retry idempotent."""
    pipe = build(spool, decision=make_decision(), publisher=FakePublisher(fail_times=1))
    pipe.handle(event)

    entry = spool.iter_pending()[0]
    entry.data["visit_id"] = 42  # insert succeeded, release did not
    entry.data.pop("next_attempt_at", None)
    spool.write_sidecar(entry.clip_path, entry.data)

    from capture.publisher import LocalDashboardPublisher

    inserts = []
    publisher = LocalDashboardPublisher(
        images_dir=spool.review_dir / "img",
        videos_dir=spool.review_dir / "vid",
        add_visit=lambda **kw: inserts.append(kw) or 99,
    )
    restarted = build(spool, publisher=publisher)
    assert restarted.drain_pending() == 1
    assert inserts == []  # no second row
    assert spool.pending_count() == 0


def test_sidecar_records_the_decision_for_later_audit(spool, event, make_decision):
    pipe = build(
        spool,
        decision=make_decision("cinnyris_indet", level="genus", confidence=0.71),
        publisher=FakePublisher(fail_times=1),
    )
    pipe.handle(event)

    data = json.loads(spool.iter_pending()[0].sidecar_path.read_text())
    assert data["label"] == "cinnyris_indet"
    assert data["taxon_level"] == "genus"
    assert data["confidence"] == pytest.approx(0.71)
    assert data["frames_scored"] == 4
