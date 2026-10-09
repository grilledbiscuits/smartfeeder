"""The keep / discard / publish decision.

Deliberately one small pure function with no I/O, no clock and no hardware, so
the rule that governs every byte written to the SD card can be tested
exhaustively off-Pi.

The rule defers entirely to `birdcam.inference.Decision.should_record` for the
"is this of interest" question. That property is the capture allowlist -- Tier
A species plus the genus fallbacks whose genus contains a Tier A target -- and
re-deriving it here would be a second copy of the taxonomy that could drift
from the first. ASSUMPTIONS.md A27 records the allowlist working correctly on
real footage (Cape Bulbul: 98.1% identified, 0.0% recorded); that behaviour
comes from `should_record` and this module must not second-guess it.

The one thing added on top is RETAIN. `uncertain` means "probably a bird, but I
cannot pin it down" (birdcam/inference.py), which is not the same as "not
interesting" -- A27 measured the deployed system missing most real birds rather
than recording squirrels, and abstained clips are the only local evidence of
what it is missing. They are kept on disk for review under their own retention
cap, and are never published: an abstention is not a visit record.
"""

from __future__ import annotations

from typing import Any

from capture.events import Outcome

# Sentinel labels from birdcam.inference. Imported by value rather than from
# the module so this file stays free of numpy at import time; the test suite
# asserts they still match.
UNCERTAIN = "uncertain"
UNKNOWN = "unknown"


def decide_outcome(
    decision: Any, *, retain_uncertain: bool, retain_unknown: bool = False
) -> Outcome:
    """Map one voted Decision to what happens to the clip on disk.

    Parameters
    ----------
    decision:
        A `birdcam.inference.Decision`, normally the output of
        `Classifier.vote()` over the sampled frames of one clip.
    retain_uncertain:
        Config flag `publish.retain_uncertain`. When False the service behaves
        exactly as the plain rule states: anything not on the allowlist is
        deleted immediately.
    retain_unknown:
        Config flag `publish.retain_unknown`, default False. Keeps clips the
        open-set failsafe rejected, which are normally deleted on sight.

        For COMMISSIONING, not for steady state. The argument against retaining
        them (below) assumes a PIR on continuous footage where unknowns are most
        of the timeline. Under the ToF trigger they are not: 2026-10-06 produced
        14 clips in 3.5 hours, every one rejected as unknown with novelty -3.81
        to -4.80 against a -4.9434 threshold -- and because they were deleted on
        sight there was no way to tell a real bird the gate had misjudged from a
        gust of wind. Retaining them is what makes the threshold tunable against
        the deployed view rather than against the old phone footage it was fitted
        on. Turn it off once the gate is trusted.
    """
    if decision is None:
        # No classification happened -- a classifier failure, not a verdict of
        # "uninteresting". Keeping the clip is the safe direction: a retained
        # clip costs disk, a deleted one is unrecoverable.
        return Outcome.RETAIN

    if getattr(decision, "should_record", False):
        return Outcome.PUBLISH

    # `unknown` is the open-set failsafe firing: not a thing the model knows
    # about. Never retained even when retain_uncertain is on -- these are the
    # squirrels, the rain and the empty feeder, and they are the bulk of the
    # timeline (A27: 60.8% of uncut frames).
    if getattr(decision, "is_unknown", False):
        return Outcome.RETAIN if retain_unknown else Outcome.DISCARD

    if retain_uncertain and getattr(decision, "label", None) == UNCERTAIN:
        return Outcome.RETAIN

    return Outcome.DISCARD


def describe(decision: Any, outcome: Outcome) -> str:
    """One-line human summary for the log."""
    if decision is None:
        return f"{outcome.value}: not classified"
    label = getattr(decision, "label", "?")
    level = getattr(decision, "level", "?")
    conf = getattr(decision, "confidence", 0.0)
    target = getattr(decision, "is_capture_target", False)
    return (
        f"{outcome.value}: {label} ({level}) conf={conf:.3f} "
        f"allowlisted={target} unknown={getattr(decision, 'is_unknown', False)}"
    )
