"""Cooldown and concurrency admission."""

from __future__ import annotations

import pytest

from capture.events import MotionEvent, Trigger
from capture.service import Admission, TriggerGate


@pytest.fixture
def clock():
    return [1000.0]


def gate(clock, **kwargs):
    params = {"cooldown_seconds": 30.0, "on_busy": "queue", "max_queued": 1}
    params.update(kwargs)
    return TriggerGate(monotonic=lambda: clock[0], **params)


def fire(g) -> Admission:
    return g.offer(MotionEvent.now(Trigger.MOCK))


def test_first_trigger_is_always_admitted(clock):
    """No cooldown window exists before the first event."""
    assert fire(gate(clock)) is Admission.ACCEPTED


def test_retrigger_inside_the_cooldown_is_dropped(clock):
    """One physical visit must not spawn overlapping recordings.

    The HC-SR501 re-asserts its output after its own hardware retrigger period
    while the same bird is still at the feeder; this is what absorbs that.
    """
    g = gate(clock)
    assert fire(g) is Admission.ACCEPTED
    g.next_event(timeout=0.01)
    g.done()
    for offset in (0.5, 5.0, 29.9):
        clock[0] = 1000.0 + offset
        assert fire(g) is Admission.COOLDOWN
    assert g.counts["dropped_cooldown"] == 3


def test_trigger_after_the_cooldown_is_admitted(clock):
    g = gate(clock)
    fire(g)
    g.next_event(timeout=0.01)
    g.done()
    clock[0] = 1030.0
    assert fire(g) is Admission.ACCEPTED


def test_busy_with_drop_policy_discards_the_trigger(clock):
    g = gate(clock, on_busy="drop")
    fire(g)
    g.next_event(timeout=0.01)  # worker picks it up -> busy
    clock[0] = 1100.0
    assert fire(g) is Admission.BUSY_DROPPED
    assert g.queued == 0


def test_busy_with_queue_policy_holds_one_then_rejects(clock):
    g = gate(clock, on_busy="queue", max_queued=1)
    fire(g)
    g.next_event(timeout=0.01)  # in flight
    clock[0] = 1100.0
    assert fire(g) is Admission.ACCEPTED  # waits behind the one in flight
    assert g.queued == 1
    clock[0] = 1200.0
    assert fire(g) is Admission.QUEUE_FULL
    assert g.counts["dropped_queue_full"] == 1


def test_queue_drains_in_order(clock):
    g = gate(clock, max_queued=3)
    first = MotionEvent.now(Trigger.MOCK)
    g.offer(first)
    got = g.next_event(timeout=0.01)
    assert got is first
    clock[0] = 1100.0
    second = MotionEvent.now(Trigger.MOCK)
    g.offer(second)
    g.done()
    assert g.next_event(timeout=0.01) is second


def test_next_event_returns_none_when_idle(clock):
    assert gate(clock).next_event(timeout=0.01) is None


def test_counts_cover_every_admission_kind(clock):
    assert set(gate(clock).counts) == {a.value for a in Admission}
