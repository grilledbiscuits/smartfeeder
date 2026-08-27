"""The keep / discard / publish rule.

This is the rule that governs every byte written to the card, so it is tested
against the real Decision type and every branch it can take.
"""

from __future__ import annotations

import pytest

from capture.events import Outcome
from capture.policy import decide_outcome


def test_allowlisted_species_is_published(make_decision):
    d = make_decision("cinnyris_chalybeus", is_capture_target=True)
    assert d.should_record is True
    assert decide_outcome(d, retain_uncertain=True) is Outcome.PUBLISH
    assert decide_outcome(d, retain_uncertain=False) is Outcome.PUBLISH


def test_genus_fallback_on_the_allowlist_is_published(make_decision):
    # `cinnyris_indet` is on the allowlist by design -- the genus contains Tier
    # A targets, so "one of the double-collareds" is worth recording.
    d = make_decision("cinnyris_indet", level="genus", is_capture_target=True)
    assert decide_outcome(d, retain_uncertain=False) is Outcome.PUBLISH


@pytest.mark.parametrize(
    "label",
    ["pycnonotus_capensis", "nectariniidae_indet", "empty_feeder", "insect", "other_animal"],
)
def test_non_allowlisted_labels_are_discarded(make_decision, label):
    """A Tier C bird, a too-vague fallback and the negatives all go.

    ASSUMPTIONS.md A27 measured this working on real footage: Cape Bulbul was
    98.1% correctly identified and 0.0% recorded.
    """
    d = make_decision(label, is_capture_target=False)
    assert decide_outcome(d, retain_uncertain=True) is Outcome.DISCARD
    assert decide_outcome(d, retain_uncertain=False) is Outcome.DISCARD


def test_unknown_is_always_discarded_even_when_retaining(make_decision):
    """The open-set failsafe firing means squirrels, rain, empty feeder.

    A27: 60.8% of uncut frames are flagged unknown. Retaining those would fill
    the card with the one category that is definitely not worth review.
    """
    d = make_decision("unknown", level="unknown", is_unknown=True, is_capture_target=False)
    assert decide_outcome(d, retain_uncertain=True) is Outcome.DISCARD


def test_uncertain_is_retained_when_configured(make_decision):
    d = make_decision("uncertain", level="uncertain", confidence=0.31, is_capture_target=False)
    assert decide_outcome(d, retain_uncertain=True) is Outcome.RETAIN
    assert decide_outcome(d, retain_uncertain=False) is Outcome.DISCARD


def test_unclassified_clip_is_retained_not_deleted():
    """No verdict is not a verdict of 'uninteresting'.

    A retained clip costs disk; a deleted one is unrecoverable.
    """
    assert decide_outcome(None, retain_uncertain=False) is Outcome.RETAIN


def test_capture_target_flag_alone_decides_not_the_level(make_decision):
    """A confident species-level result that is NOT on the allowlist still goes.

    This is the regression the `should_record` fix was for: recording is a
    question about WHICH taxon, not how specific the answer is.
    """
    confident_bystander = make_decision(
        "pycnonotus_capensis", level="species", confidence=0.98, is_capture_target=False
    )
    assert decide_outcome(confident_bystander, retain_uncertain=True) is Outcome.DISCARD
