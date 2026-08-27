"""Config loading, env expansion, and the cross-checks that must be fatal."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from capture.config import CaptureConfig, CaptureConfigError, expand_env

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "capture.example.yaml"


def write(tmp_path, mutate=None) -> Path:
    data = yaml.safe_load(EXAMPLE.read_text())
    if mutate:
        mutate(data)
    path = tmp_path / "capture.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_the_shipped_example_is_valid(tmp_path):
    """The reference config must actually load; it is what operators copy."""
    cfg = CaptureConfig.load(write(tmp_path), root=tmp_path)
    assert cfg.get("capture.clip_seconds") == 8.0
    assert cfg.get("gpio.pin") == 4


def test_missing_key_names_the_key(tmp_path):
    path = write(tmp_path, lambda d: d["capture"].pop("cooldown_seconds"))
    with pytest.raises(CaptureConfigError, match="capture.cooldown_seconds"):
        CaptureConfig.load(path, root=tmp_path)


def test_missing_section_is_reported(tmp_path):
    path = write(tmp_path, lambda d: d.pop("storage"))
    with pytest.raises(CaptureConfigError, match="'storage' is missing"):
        CaptureConfig.load(path, root=tmp_path)


def test_cooldown_shorter_than_the_clip_is_fatal(tmp_path):
    """Otherwise a second trigger starts while the first clip is still writing."""

    def shrink(d):
        d["capture"]["cooldown_seconds"] = 4.0
        d["capture"]["clip_seconds"] = 8.0

    with pytest.raises(CaptureConfigError, match="shorter than"):
        CaptureConfig.load(write(tmp_path, shrink), root=tmp_path)


def test_odd_video_dimensions_are_rejected(tmp_path):
    """H.264 4:2:0 needs even dimensions; the encoder's own error names nothing."""
    path = write(tmp_path, lambda d: d["camera"].update(width=1281))
    with pytest.raises(CaptureConfigError, match="even"):
        CaptureConfig.load(path, root=tmp_path)


def test_queue_policy_with_zero_depth_is_rejected(tmp_path):
    def mutate(d):
        d["capture"]["on_busy"] = "queue"
        d["capture"]["max_queued"] = 0

    with pytest.raises(CaptureConfigError, match="on_busy: drop"):
        CaptureConfig.load(write(tmp_path, mutate), root=tmp_path)


def test_unknown_busy_policy_is_rejected(tmp_path):
    path = write(tmp_path, lambda d: d["capture"].update(on_busy="explode"))
    with pytest.raises(CaptureConfigError, match="on_busy"):
        CaptureConfig.load(path, root=tmp_path)


def test_pending_cap_that_could_outgrow_the_free_space_floor_is_rejected(tmp_path):
    def mutate(d):
        d["storage"]["max_pending_clips"] = 100000
        d["storage"]["min_free_mb"] = 10

    with pytest.raises(CaptureConfigError, match="undeletable"):
        CaptureConfig.load(write(tmp_path, mutate), root=tmp_path)


def test_backoff_that_shrinks_is_rejected(tmp_path):
    path = write(tmp_path, lambda d: d["publish"].update(backoff_factor=0.5))
    with pytest.raises(CaptureConfigError, match="backoff_factor"):
        CaptureConfig.load(path, root=tmp_path)


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("BIRDCAM_TEST_TOKEN", "s3cret")
    assert expand_env("${BIRDCAM_TEST_TOKEN}") == "s3cret"
    assert expand_env({"a": ["${BIRDCAM_TEST_TOKEN}"]}) == {"a": ["s3cret"]}


def test_env_fallback_is_used_when_unset(monkeypatch):
    monkeypatch.delenv("BIRDCAM_ABSENT", raising=False)
    assert expand_env("${BIRDCAM_ABSENT:-default}") == "default"


def test_unset_env_without_fallback_is_fatal(monkeypatch):
    """An empty string here would silently authenticate as nobody."""
    monkeypatch.delenv("BIRDCAM_ABSENT", raising=False)
    with pytest.raises(CaptureConfigError, match="BIRDCAM_ABSENT"):
        expand_env("${BIRDCAM_ABSENT}")


def test_relative_paths_resolve_against_the_repo_root(tmp_path):
    cfg = CaptureConfig.load(write(tmp_path), root=tmp_path)
    assert cfg.resolve_path("storage.work_dir") == tmp_path / "var" / "capture" / "work"
