"""Field frames as a training source.

These frames are the project's only in-domain data and its only negatives, and
they are also the easiest data in the corpus to leak. Two clips from one
recording session are the same bird in the same light minutes apart, and the
empty-feeder frames mined from that session's gaps share its background exactly.
Split those across train and test and every metric becomes fiction in a way that
looks like success.
"""

from __future__ import annotations

import collections

import pytest

from birdcam.config import load_config
from birdcam.data.field_source import load_field


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def field(cfg):
    items = load_field(cfg)
    if not items:
        pytest.skip("no extracted field frames on this machine")
    return items


def test_no_session_spans_two_splits(field) -> None:
    """The leak this whole module is arranged to prevent."""
    splits = collections.defaultdict(set)
    for it in field:
        splits[it.observer_id].add(it.split)
    straddling = {s: v for s, v in splits.items() if len(v) > 1}
    assert not straddling, f"sessions in more than one split: {straddling}"


def test_every_split_is_populated(field) -> None:
    counts = collections.Counter(it.split for it in field)
    for s in ("train", "val", "test"):
        assert counts[s] > 0, f"{s} split is empty"


def test_splits_are_roughly_on_target(field, cfg) -> None:
    """Sessions are chunky, so exactness is impossible; gross skew is not.

    Placement is greedy on frames rather than session count precisely because
    counting sessions equally gave 75/20/5 against a 70/15/15 target.
    """
    counts = collections.Counter(it.split for it in field)
    n = len(field)
    assert 0.6 <= counts["train"] / n <= 0.8
    assert counts["val"] / n >= 0.08
    assert counts["test"] / n >= 0.08


def test_multi_bird_frames_are_excluded(field) -> None:
    """Two birds in frame: no single taxon is true of it, so it is not usable."""
    assert not [it for it in field if "multibird" in str(it.path)]


def test_only_mined_uncut_frames_appear(cfg, field) -> None:
    """Uncut footage is mostly unlabelled; only mined negatives may be used."""
    for it in field:
        if "/uncut/" in str(it.path):
            assert it.taxon_label == "empty_feeder", f"{it.path} entered as {it.taxon_label}"


def test_field_frames_are_weighted_below_web(field, cfg) -> None:
    """A field frame must not count as much as a web image -- see field_source."""
    for it in field:
        assert it.source == "field"
        assert 0.0 < it.weight < 1.0


def test_every_frame_exists_on_disk(field) -> None:
    missing = [it.path for it in field if not it.path.is_file()]
    assert not missing, f"{len(missing)} field frames are indexed but absent"


def test_labels_are_real_classes(field, cfg) -> None:
    for it in field:
        assert cfg.taxon_classes[it.taxon_index] == it.taxon_label


def test_adding_a_session_does_not_reshuffle_others(cfg) -> None:
    """Per-label seeding: a new session must not move an unrelated label's.

    The web splits had exactly this bug -- one shared rng consumed in sorted
    order, so adding the drongo reassigned every species sorting after it.
    """
    from birdcam.data.field_source import assign_sessions

    sessions = {"s1": "a", "s2": "a", "s3": "a", "s4": "b", "s5": "b", "s6": "b"}
    sizes = dict.fromkeys(sessions, 100)
    before = assign_sessions(sessions, sizes, cfg)

    sessions2 = {**sessions, "s7": "c"}
    sizes2 = {**sizes, "s7": 100}
    after = assign_sessions(sessions2, sizes2, cfg)

    for s in sessions:
        assert before[s] == after[s], f"{s} moved from {before[s]} to {after[s]}"
