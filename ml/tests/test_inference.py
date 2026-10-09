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


def distribute(cfg, clf, weights: dict[str, float]):
    """Decide on an explicit probability distribution over species.

    This is how a rollup actually happens in production: no single species
    clears its own threshold, but the group they belong to does. It has to be
    exercised this way because the rollup labels are NOT model outputs -- there
    is no logit to force. `decide()` builds them from summed species mass.

    Weights are normalised, then turned back into logits. Note that Tier A
    species carry per-class thresholds as low as 0.18, so "split it evenly over
    two species" is not enough to prevent one of them simply winning.
    """
    idx = cfg.taxon_class_index
    total = sum(weights.values())
    probs = np.full(len(cfg.taxon_classes), 1e-12)
    for slug, w in weights.items():
        probs[idx[slug]] = w / total
    return clf.decide(np.log(probs), np.zeros(len(cfg.sex_classes)), features=None)


def _slugs_in_family(cfg, family: str) -> list[str]:
    return [s.slug for s in cfg.species if cfg.genus_to_family.get(s.genus) == family]


def _even(slugs: list[str]) -> dict[str, float]:
    return dict.fromkeys(slugs, 1.0)


def decide_for(cfg, clf, label: str):
    """Force `label` to win the argmax with overwhelming confidence."""
    idx = cfg.taxon_class_index
    assert label in idx, f"{label} is not a configured taxon class"
    z = np.full(len(cfg.taxon_classes), -10.0)
    z[idx[label]] = 10.0
    return clf.decide(z, np.zeros(len(cfg.sex_classes)), features=None)


# --- negatives must never record ----------------------------------------------


@pytest.mark.parametrize("label", ["other_animal"])
def test_negative_classes_never_record(cfg, clf, label) -> None:
    """The regression that motivated this file.

    `insect` and `obstruction` are no longer outputs: there is no training data
    for either, and an output that never sees an example can still win an
    argmax. `empty_feeder` is an output but is suppressed before anything reads
    the probabilities -- see the test below. All three are handled off the
    classifier now.
    """
    d = decide_for(cfg, clf, label)
    assert d.level == "negative", f"{label} reported level {d.level!r}"
    assert not d.should_record


def test_negative_class_is_not_called_a_species(cfg, clf) -> None:
    """A confident 'nothing here' is not a species identification."""
    assert decide_for(cfg, clf, "other_animal").level != "species"


def test_empty_feeder_can_never_be_emitted(cfg, clf) -> None:
    """Measured 2026-09-11: the class does not work and is harmful.

    Recall 1.000 on the nine backgrounds it trained on, 0.000 on one it had not
    seen -- it memorised backgrounds, which is the obvious shortcut when the
    class is defined by the absence of a subject. All seven of its validation
    predictions were false positives on real birds, including an Amethyst
    Sunbird, each of which it would have declined to record.

    So it is masked before the softmax. Emptiness is decided geometrically from
    a rolling background instead: capture/emptygate.py. The output remains in
    the head only so the trained checkpoint stays loadable.
    """
    d = decide_for(cfg, clf, "empty_feeder")
    assert d.label != "empty_feeder"
    assert not any(k == "empty_feeder" for k, _ in d.top_k)
    assert not d.should_record


def test_suppressed_classes_renormalise_rather_than_leak_mass(cfg, clf) -> None:
    """Masking before the softmax, not zeroing after it.

    Zeroing afterwards would leave the probabilities summing to less than one,
    and every rollup threshold downstream would read low.
    """
    idx = cfg.taxon_class_index
    z = np.full(len(cfg.taxon_classes), -10.0)
    z[idx["empty_feeder"]] = 10.0
    z[idx["cinnyris_chalybeus"]] = 9.0
    d = clf.decide(z, np.zeros(len(cfg.sex_classes)), features=None)
    assert d.label == "cinnyris_chalybeus"
    assert d.confidence > 0.9, "mass did not renormalise onto the surviving classes"


# --- fallback nodes describe their own generality -----------------------------


def test_genus_fallback_reports_genus_level(cfg, clf) -> None:
    """Spread thinly over Cinnyris: no species wins, the genus does."""
    d = distribute(cfg, clf, _even([s.slug for s in cfg.species if s.genus == "Cinnyris"]))
    assert d.label == "cinnyris_indet"
    assert d.level == "genus"


def test_family_fallback_reports_family_level(cfg, clf) -> None:
    """Spread across the whole sunbird family: no genus reaches 0.65, the family does."""
    d = distribute(cfg, clf, _even(_slugs_in_family(cfg, "Nectariniidae")))
    assert d.label == "nectariniidae_indet"
    assert d.level == "family"


def test_guild_fallback_reports_guild_level(cfg, clf) -> None:
    """Sunbirds and sugarbirds together: neither family clears, the guild does.

    The sugarbird carries most of the weight so it takes the argmax -- the
    rollup walks up from the winning class -- but stays under its own 0.18
    threshold, so it cannot simply win as a species.
    """
    w = _even(_slugs_in_family(cfg, "Nectariniidae") + _slugs_in_family(cfg, "Promeropidae"))
    w["promerops_cafer"] = 2.5
    d = distribute(cfg, clf, w)
    assert d.label == "nectarivore_indet"
    assert d.level == "guild"


def test_rollup_nodes_are_not_model_outputs(cfg) -> None:
    """They describe a distribution, not a photographable thing.

    Every one of them held zero training images. An untrained output still
    produces a logit and can still win an argmax, which would report a
    confident identification with nothing behind it.
    """
    assert not [c for c in cfg.taxon_classes if c.endswith("_indet")]


def test_every_class_in_the_head_is_trainable(cfg) -> None:
    """Species, plus only those negatives real data exists for."""
    head = cfg.taxonomy_cfg["taxon_head"]
    expected = {s.slug for s in cfg.species} | set(head["trainable_negatives"])
    assert set(cfg.taxon_classes) == expected


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
    d = distribute(cfg, clf, _even([s.slug for s in cfg.species if s.genus == "Cinnyris"]))
    assert d.label == "cinnyris_indet"
    assert d.should_record


def test_target_bearing_family_fallback_records(cfg, clf) -> None:
    """'Some sunbird' records: female sunbirds often resolve no further."""
    d = distribute(cfg, clf, _even(_slugs_in_family(cfg, "Nectariniidae")))
    assert d.label == "nectariniidae_indet"
    assert d.should_record


def test_non_target_family_fallback_does_not_record(cfg, clf) -> None:
    assert "zosteropidae_indet" not in clf._capture_targets
    assert "dicruridae_indet" not in clf._capture_targets


def test_guild_fallback_does_not_record(cfg, clf) -> None:
    """'Some nectarivore' is too vague to justify storage."""
    w = _even(_slugs_in_family(cfg, "Nectariniidae") + _slugs_in_family(cfg, "Promeropidae"))
    w["promerops_cafer"] = 2.5
    d = distribute(cfg, clf, w)
    assert d.level == "guild"
    assert not d.should_record


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


@pytest.mark.parametrize(
    "confidence, expected",
    [(0.819, "uncertain"), (0.844, "uncertain"), (0.94, "nectariniidae_indet")],
)
def test_sunbird_family_vote_requires_higher_confidence(clf, confidence, expected) -> None:
    from birdcam.inference import Decision

    frames = [
        Decision("nectariniidae_indet", "family", confidence, is_capture_target=True)
        for _ in range(3)
    ]
    voted = clf.vote(frames)
    assert voted.label == expected
    assert voted.should_record == (expected == "nectariniidae_indet")


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
