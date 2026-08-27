"""The dashboard interface contract: media into var/media/, one row via add_visit."""

from __future__ import annotations

import pytest

from capture.events import SpoolEntry
from capture.publisher import TIMESTAMP_FORMAT, LocalDashboardPublisher, PublishError


@pytest.fixture
def publisher(tmp_path):
    calls = []

    def add_visit(**kwargs):
        calls.append(kwargs)
        return len(calls)

    pub = LocalDashboardPublisher(
        images_dir=tmp_path / "media" / "images",
        videos_dir=tmp_path / "media" / "videos",
        add_visit=add_visit,
    )
    pub.calls = calls
    return pub


def make_entry(tmp_path, **overrides) -> SpoolEntry:
    clip = tmp_path / "pending" / "evt1.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00" * 256)
    data = {
        "event_id": "evt1",
        "label": "cinnyris_chalybeus",
        "confidence": 0.8123,
        "detected_at": "2026-08-27T06:15:00",
        "duration_s": 8.0,
        "keyframe": None,
    }
    data.update(overrides)
    return SpoolEntry(clip_path=clip, sidecar_path=clip.with_suffix(".json"), data=data)


def test_media_is_written_before_the_row(tmp_path, publisher):
    entry = make_entry(tmp_path)
    visit_id = publisher.publish(entry)

    assert visit_id == 1
    assert (publisher.videos_dir / "evt1.mp4").is_file()
    call = publisher.calls[0]
    assert call["video_filename"] == "evt1.mp4"
    assert call["confidence"] == pytest.approx(0.8123)
    assert call["duration_seconds"] == 8.0


def test_keyframe_becomes_the_thumbnail(tmp_path, publisher):
    entry = make_entry(tmp_path, keyframe="evt1.jpg")
    (entry.clip_path.parent / "evt1.jpg").write_bytes(b"\xff\xd8\xff")

    publisher.publish(entry)
    assert (publisher.images_dir / "evt1.jpg").is_file()
    assert publisher.calls[0]["image_filename"] == "evt1.jpg"


def test_missing_keyframe_still_publishes(tmp_path, publisher):
    """The dashboard renders 'no image' rather than breaking."""
    publisher.publish(make_entry(tmp_path))
    assert publisher.calls[0]["image_filename"] is None


def test_timestamp_matches_the_existing_row_format(tmp_path, publisher):
    """The dashboard orders by the raw TEXT column.

    An ISO 'T' separator would sort every new row incorrectly against the rows
    written by web/scripts/create_dummy_data.py, because ' ' < 'T'.
    """
    from datetime import datetime

    publisher.publish(make_entry(tmp_path))
    stamp = publisher.calls[0]["timestamp"]
    assert datetime.strptime(stamp, TIMESTAMP_FORMAT)
    assert stamp == "2026-08-27 06:15:00"


def test_slug_is_rendered_as_a_common_name_when_config_is_present(tmp_path):
    class FakeSpecies:
        slug = "cinnyris_chalybeus"
        common_name = "Southern Double-collared Sunbird"
        scientific_name = "Cinnyris chalybeus"

    class FakeConfig:
        species = [FakeSpecies()]

    calls = []
    pub = LocalDashboardPublisher(
        images_dir=tmp_path / "i",
        videos_dir=tmp_path / "v",
        add_visit=lambda **kw: calls.append(kw) or 1,
        birdcam_config=FakeConfig(),
    )
    pub.publish(make_entry(tmp_path))
    assert calls[0]["species"] == "Southern Double-collared Sunbird"


def test_genus_fallback_never_claims_a_species(tmp_path, publisher):
    """`cinnyris_indet` stands for two species that cannot be told apart."""
    publisher.publish(make_entry(tmp_path, label="cinnyris_indet"))
    assert publisher.calls[0]["species"] == "Cinnyris sp."


def test_publish_is_idempotent_once_a_visit_id_exists(tmp_path, publisher):
    entry = make_entry(tmp_path, visit_id=77)
    assert publisher.publish(entry) == 77
    assert publisher.calls == []


def test_missing_clip_raises_publish_error(tmp_path, publisher):
    entry = make_entry(tmp_path)
    entry.clip_path.unlink()
    with pytest.raises(PublishError, match="gone from the spool"):
        publisher.publish(entry)


def test_add_visit_failure_becomes_publish_error(tmp_path):
    def boom(**kwargs):
        raise RuntimeError("database is locked")

    pub = LocalDashboardPublisher(
        images_dir=tmp_path / "i", videos_dir=tmp_path / "v", add_visit=boom
    )
    with pytest.raises(PublishError, match="database is locked"):
        pub.publish(make_entry(tmp_path))
