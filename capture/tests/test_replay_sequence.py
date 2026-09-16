"""Replay verdicts must reflect retained outcomes, including after-record gating."""

from types import SimpleNamespace

from capture.events import Outcome
from deploy.replay_sequence import is_correct


def test_after_record_empty_discard_is_success():
    record = SimpleNamespace(
        error=None, decision=None, clip_path="recorded.mp4", empty=True, outcome=Outcome.DISCARD
    )
    assert is_correct("empty_feeder", record)[0]
    assert not is_correct("cinnyris_chalybeus", record)[0]


def test_recording_error_cannot_pass_as_empty():
    record = SimpleNamespace(
        error="camera failed", decision=None, clip_path=None, empty=False, outcome=None
    )
    assert not is_correct("empty_feeder", record)[0]
