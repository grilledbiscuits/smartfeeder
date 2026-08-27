"""Tests for the capture decision.

These exist because the decision was wrong in a way no existing test noticed.
`decide()` labelled every winning class "species" regardless of what it was,
and `should_record` keyed off that level, so `empty_feeder` and `insect` --
classes whose entire purpose is to suppress recording -- returned
`should_record=True`. An empty feeder would have triggered a capture.

The lesson generalises: a level tells you how *specific* an answer is, never
whether it is worth acting on. Every test here asserts on the taxon, not the
level.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdcam.config import load_config
from birdcam.inference import Classifier


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def clf(cfg):
    return Classifier(cfg)


def decide_for(cfg, clf, label: str):
    """Force `label` to win the argmax with overwhelming confidence."""
    idx = cfg.taxon_class_index
    assert label in idx, f"{label} is not a configured taxon class"
    z = np.full(len(cfg.taxon_classes), -10.0)
    z[idx[label]] = 10.0
    return clf.decide(z, np.zeros(len(cfg.sex_classes)), features=None)


# --- negatives must never record ----------------------------------------------


@pytest.mark.parametrize("label", ["empty_feeder", "insect", "other_animal", "obstruction"])
def test_negative_classes_never_record(cfg, clf, label) -> None:
    """The regression that motivated this file."""
    d = decide_for(cfg, clf, label)
    assert d.level == "negative", f"{label} reported level {d.level!r}"
    assert not d.should_record


def test_negative_class_is_not_called_a_species(cfg, clf) -> None:
    """A confident 'nothing here' is not a species identification."""
    assert decide_for(cfg, clf, "empty_feeder").level != "species"


# --- fallback nodes describe their own generality -----------------------------


def test_genus_fallback_reports_genus_level(cfg, clf) -> None:
    d = decide_for(cfg, clf, "cinnyris_indet")
    assert d.level == "genus"


def test_family_fallback_reports_family_level(cfg, clf) -> None:
    assert decide_for(cfg, clf, "nectariniidae_indet").level == "family"


def test_guild_fallback_reports_guild_level(cfg, clf) -> None:
    assert decide_for(cfg, clf, "nectarivore_indet").level == "guild"


# --- the capture allowlist ----------------------------------------------------


def test_every_tier_a_species_records(cfg, clf) -> None:
    for s in cfg.species_by_tier("A"):
        d = decide_for(cfg, clf, s.slug)
        assert d.should_record, f"Tier A {s.slug} would not be recorded"


def test_non_target_species_is_identified_but_not_recorded(cfg, clf) -> None:
    """Tier C birds are hard negatives: recognised, deliberately not stored.

    Cape White-eye is a real visitor and a real confuser -- 16% of
    C. chalybeus test errors go to it -- so the model must be able to name it
    without that naming committing video.
    """
    d = decide_for(cfg, clf, "zosterops_virens")
    assert d.level == "species"
    assert not d.should_record


def test_target_bearing_genus_fallback_records(cfg, clf) -> None:
    """'One of the double-collared sunbirds' is worth recording; both are targets."""
    assert decide_for(cfg, clf, "cinnyris_indet").should_record


def test_vague_fallbacks_do_not_record(cfg, clf) -> None:
    """'Some sunbird' is too vague to justify storage."""
    assert not decide_for(cfg, clf, "nectariniidae_indet").should_record
    assert not decide_for(cfg, clf, "nectarivore_indet").should_record


# --- the unknown gate still wins ----------------------------------------------


def test_unknown_suppresses_a_target(cfg, clf) -> None:
    """Novelty short-circuits everything: an OOD frame records nothing."""

    class _AlwaysNovel:
        threshold = 0.0

        def score(self, feats, logits=None):
            return np.array([1e9])

    c = Classifier(cfg, novelty_scorer=_AlwaysNovel())
    idx = cfg.taxon_class_index["cinnyris_chalybeus"]
    z = np.full(len(cfg.taxon_classes), -10.0)
    z[idx] = 10.0
    d = c.decide(z, np.zeros(len(cfg.sex_classes)), features=np.zeros((1, 8)))
    assert d.is_unknown
    assert not d.should_record


# --- the vote must preserve what the frames decided ----------------------------
#
# The capture application never acts on a single frame: it samples a clip,
# calls decide() per frame and then vote(), and reads should_record off the
# VOTED decision. vote() rebuilt the Decision field by field and omitted
# is_capture_target, so the flag reverted to its default of False and every
# voted decision -- including a unanimous, confident Tier A target -- was
# discarded. Every test above passes on the per-frame path and none of them
# touched this one.


def test_vote_preserves_capture_target(cfg, clf) -> None:
    """A unanimous Tier A vote must still be recordable."""
    frames = [decide_for(cfg, clf, "cinnyris_chalybeus") for _ in range(3)]
    assert all(f.should_record for f in frames)

    voted = clf.vote(frames)
    assert voted.label == "cinnyris_chalybeus"
    assert voted.is_capture_target
    assert voted.should_record, "a voted Tier A target would not be recorded"


def test_vote_does_not_invent_a_capture_target(cfg, clf) -> None:
    """The flag is carried, not re-derived: a Tier C vote stays unrecordable."""
    frames = [decide_for(cfg, clf, "zosterops_virens") for _ in range(3)]
    voted = clf.vote(frames)
    assert voted.label == "zosterops_virens"
    assert not voted.is_capture_target
    assert not voted.should_record


def test_vote_over_every_tier_a_species_records(cfg, clf) -> None:
    for s in cfg.species_by_tier("A"):
        voted = clf.vote([decide_for(cfg, clf, s.slug) for _ in range(3)])
        assert voted.should_record, f"voted Tier A {s.slug} would not be recorded"


def test_majority_unknown_still_suppresses_a_target(cfg, clf) -> None:
    """The unknown branch has no members to carry a flag; it must stay off."""
    from birdcam.inference import Decision

    frames = [
        Decision(label="unknown", level="unknown", confidence=0.0, is_unknown=True),
        Decision(label="unknown", level="unknown", confidence=0.0, is_unknown=True),
        decide_for(cfg, clf, "cinnyris_chalybeus"),
    ]
    voted = clf.vote(frames)
    assert voted.is_unknown
    assert not voted.should_record
