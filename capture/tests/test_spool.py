"""Durable queue behaviour and the storage guards.

The single invariant under test throughout: a clip whose publication is
unconfirmed is never deleted, for any reason, including to make room.
"""

from __future__ import annotations

import json

import pytest

from capture.events import CaptureRecord, MotionEvent, Trigger
from capture.spool import InsufficientStorage


def record_with_clip(spool, name: str = "evt", *, keyframe: bool = False) -> CaptureRecord:
    event = MotionEvent(
        event_id=name, detected_at=MotionEvent.now().detected_at, trigger=Trigger.MOCK
    )
    rec = CaptureRecord(event=event, duration_seconds=8.0)
    rec.clip_path = spool.work_path(name)
    rec.clip_path.write_bytes(b"\x00" * 1024)
    if keyframe:
        rec.keyframe_path = spool.work_dir / f"{name}.jpg"
        rec.keyframe_path.write_bytes(b"\xff\xd8\xff")
    return rec


def test_enqueue_moves_clip_and_writes_sidecar(spool):
    rec = record_with_clip(spool, "e1", keyframe=True)
    entry = spool.enqueue(rec)

    assert entry.clip_path.parent == spool.pending_dir
    assert entry.clip_path.is_file()
    assert entry.sidecar_path.is_file()
    assert entry.keyframe_path.is_file()
    assert not (spool.work_dir / "e1.mp4").exists()

    data = json.loads(entry.sidecar_path.read_text())
    assert data["event_id"] == "e1"
    assert data["attempts"] == 0
    assert data["visit_id"] is None


def test_iter_pending_readopts_a_clip_with_no_sidecar(spool):
    """A parse error is not a reason to destroy footage."""
    orphan = spool.pending_dir / "orphan.mp4"
    orphan.write_bytes(b"\x00" * 512)

    entries = spool.iter_pending()
    assert len(entries) == 1
    assert entries[0].event_id == "orphan"
    assert entries[0].data["recovered"] is True
    assert orphan.is_file()


def test_iter_pending_readopts_a_corrupt_sidecar(spool):
    clip = spool.pending_dir / "bad.mp4"
    clip.write_bytes(b"\x00" * 512)
    clip.with_suffix(".json").write_text("{not json")

    entries = spool.iter_pending()
    assert len(entries) == 1
    assert clip.is_file()


def test_release_refuses_without_a_confirmed_visit_id(spool):
    entry = spool.enqueue(record_with_clip(spool, "e2"))
    with pytest.raises(ValueError, match="unconfirmed"):
        spool.release(entry, delete=True)
    assert entry.clip_path.is_file()


def test_release_deletes_only_after_confirmation(spool):
    entry = spool.enqueue(record_with_clip(spool, "e3", keyframe=True))
    spool.mark_published(entry, visit_id=17)
    assert json.loads(entry.sidecar_path.read_text())["visit_id"] == 17

    spool.release(entry, delete=True)
    assert not entry.clip_path.exists()
    assert not entry.sidecar_path.exists()
    assert spool.pending_count() == 0


def test_release_without_delete_moves_the_clip_out_of_pending(spool):
    """A retained local copy must stop counting against the recording cap."""
    entry = spool.enqueue(record_with_clip(spool, "e4"))
    spool.mark_published(entry, visit_id=3)
    spool.release(entry, delete=False)

    assert spool.pending_count() == 0
    assert (spool.review_dir / "e4.mp4").is_file()


def test_pending_cap_stops_recording_rather_than_deleting(spool):
    for i in range(spool.max_pending_clips):
        spool.enqueue(record_with_clip(spool, f"p{i}"))

    with pytest.raises(InsufficientStorage, match="awaiting publication"):
        spool.check_can_record()
    # Nothing was evicted to make room.
    assert spool.pending_count() == spool.max_pending_clips


def test_low_free_space_evicts_review_clips_first(spool, monkeypatch):
    for i in range(3):
        spool.retain_for_review(record_with_clip(spool, f"r{i}"))
    assert spool.review_count() == 3

    calls = {"n": 0}

    def fake_free_mb():
        # Below the floor until the review directory has been cleared.
        calls["n"] += 1
        return 0.0 if spool.review_count() else 9999.0

    monkeypatch.setattr(spool, "free_mb", fake_free_mb)
    spool.check_can_record()  # must not raise
    assert spool.review_count() == 0


def test_low_free_space_with_nothing_evictable_raises(spool, monkeypatch):
    monkeypatch.setattr(spool, "free_mb", lambda: 0.0)
    with pytest.raises(InsufficientStorage, match="not deletable"):
        spool.check_can_record()


def test_prune_review_evicts_oldest_first(spool):
    """Review clips ARE evictable, and the oldest go first."""
    import os
    import time

    now = time.time()
    for i in range(5):
        clip = spool.review_dir / f"v{i}.mp4"
        clip.write_bytes(b"\x00" * 128)
        clip.with_suffix(".json").write_text("{}")
        os.utime(clip, (now + i, now + i))

    assert spool.prune_review() == 2  # cap is 3
    remaining = sorted(p.stem for p in spool.review_dir.glob("*.mp4"))
    assert remaining == ["v2", "v3", "v4"]
    assert not (spool.review_dir / "v0.json").exists()


def test_retain_for_review_enforces_the_cap_as_it_goes(spool):
    for i in range(5):
        spool.retain_for_review(record_with_clip(spool, f"w{i}"))
    assert spool.review_count() == spool.max_review_clips


def test_clear_work_removes_crash_debris(spool):
    (spool.work_dir / "half-written.mp4").write_bytes(b"\x00")
    assert spool.clear_work() == 1
    assert not any(spool.work_dir.iterdir())


def test_discard_removes_clip_and_keyframe(spool):
    rec = record_with_clip(spool, "gone", keyframe=True)
    spool.discard(rec)
    assert not rec.clip_path.exists()
    assert not rec.keyframe_path.exists()
