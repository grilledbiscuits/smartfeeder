"""The long-running service: admission control, the worker loop, lifecycle.

Two responsibilities, kept apart:

* `TriggerGate` decides whether a motion event becomes work. It is pure logic
  over a clock and a queue, so the cooldown and the concurrency policy are
  testable without a camera or a sensor.
* `CaptureService` owns the threads, the signals and the hardware handles.

Why the cooldown is stamped at ADMISSION
----------------------------------------
The brief asks for a cooldown after a recording starts. When nothing is busy
those are the same instant. When the previous event is still classifying, they
are not -- and stamping at admission is the stricter reading: it stops a queue
filling with re-triggers from the same bird, which is exactly what the
HC-SR501's own retrigger behaviour produces. Stamping at recording start would
let a burst of sensor edges queue up and then record the same visit several
times over.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum

from capture.events import MotionEvent
from capture.logging_setup import log_event

logger = logging.getLogger(__name__)


class Admission(StrEnum):
    ACCEPTED = "accepted"
    COOLDOWN = "dropped_cooldown"
    BUSY_DROPPED = "dropped_busy"
    QUEUE_FULL = "dropped_queue_full"


class TriggerGate:
    """Cooldown, concurrency policy and the bounded work queue."""

    def __init__(
        self,
        *,
        cooldown_seconds: float,
        on_busy: str = "queue",
        max_queued: int = 1,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cooldown = float(cooldown_seconds)
        self.on_busy = on_busy
        self.max_queued = int(max_queued)
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._queue: deque[MotionEvent] = deque()
        self._busy = False
        # -inf so the very first trigger is never inside a cooldown window.
        self._last_admit = float("-inf")
        self.counts: dict[str, int] = {a.value: 0 for a in Admission}

    def offer(self, event: MotionEvent) -> Admission:
        """Called from the sensor's thread. Never blocks."""
        with self._lock:
            now = self._monotonic()
            if now - self._last_admit < self.cooldown:
                return self._count(Admission.COOLDOWN, event, now)

            if self._busy or self._queue:
                if self.on_busy == "drop":
                    return self._count(Admission.BUSY_DROPPED, event, now)
                if len(self._queue) >= self.max_queued:
                    return self._count(Admission.QUEUE_FULL, event, now)

            self._last_admit = now
            self._queue.append(event)
            self._ready.set()
            return self._count(Admission.ACCEPTED, event, now)

    def _count(self, admission: Admission, event: MotionEvent, now: float) -> Admission:
        self.counts[admission.value] += 1
        if admission is not Admission.ACCEPTED:
            # Dropped triggers are logged, never silent: a run of them is the
            # signal that the cooldown or the queue cap is mistuned.
            log_event(
                logger,
                logging.INFO,
                "motion trigger dropped",
                {
                    "event_id": event.event_id,
                    "reason": admission.value,
                    "since_last_admit_s": (
                        None
                        if self._last_admit == float("-inf")
                        else round(now - self._last_admit, 1)
                    ),
                    "queued": len(self._queue),
                },
            )
        return admission

    def next_event(self, timeout: float = 1.0) -> MotionEvent | None:
        """Called from the worker. Returns None on timeout."""
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            event = self._queue.popleft() if self._queue else None
            if not self._queue:
                self._ready.clear()
            if event is not None:
                self._busy = True
            return event

    def done(self) -> None:
        with self._lock:
            self._busy = False

    @property
    def queued(self) -> int:
        with self._lock:
            return len(self._queue)


class CaptureService:
    """Wires a motion source to the pipeline and keeps them running."""

    def __init__(
        self,
        motion_source,
        pipeline,
        gate: TriggerGate,
        *,
        recorder=None,
        drain_interval_seconds: float = 30.0,
    ) -> None:
        self.motion_source = motion_source
        self.pipeline = pipeline
        self.gate = gate
        self.recorder = recorder
        self.drain_interval = float(drain_interval_seconds)
        self._stop = threading.Event()
        self._last_drain = 0.0

    def request_stop(self, *_args) -> None:
        logger.info("shutdown requested")
        self._stop.set()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request_stop)

    def run(self) -> int:
        """Block until stopped. Returns a process exit code."""
        # Debris from a crash is not a clip; clear it before anything counts
        # against the storage caps.
        removed = self.pipeline.spool.clear_work()
        if removed:
            logger.warning("cleared %d unfinalised file(s) from the work directory", removed)

        # A queue that outlived a reboot resumes here, before the sensor is
        # even armed.
        recovered = self.pipeline.drain_pending()
        pending = len(self.pipeline.spool.iter_pending())
        logger.info(
            "startup: %d clip(s) published from the backlog, %d still pending",
            recovered,
            pending,
        )

        try:
            self.motion_source.start(self._on_motion)
        except Exception as exc:  # noqa: BLE001 - report the reason, do not traceback at boot
            logger.error("could not start the motion source: %s", exc)
            return 1

        logger.info("armed; waiting for motion")
        try:
            self._loop()
        finally:
            self._shutdown()
        return 0

    def _on_motion(self, event: MotionEvent) -> None:
        admission = self.gate.offer(event)
        if admission is Admission.ACCEPTED:
            logger.info("%s: motion admitted (queued=%d)", event.event_id, self.gate.queued)

    def _loop(self) -> None:
        while not self._stop.is_set():
            event = self.gate.next_event(timeout=1.0)
            if event is None:
                self._maybe_drain()
                continue
            try:
                self.pipeline.handle(event)
            except Exception:  # noqa: BLE001 - the loop outlives any single event
                logger.exception("%s: unhandled error; continuing", event.event_id)
            finally:
                self.gate.done()
            self._maybe_drain(force=True)

    def _maybe_drain(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_drain < self.drain_interval:
            return
        self._last_drain = now
        try:
            self.pipeline.drain_pending()
        except Exception:  # noqa: BLE001 - a stuck queue must not kill the loop
            logger.exception("draining the pending queue failed; will retry")

    def _shutdown(self) -> None:
        try:
            self.motion_source.stop()
        except Exception:  # noqa: BLE001
            logger.exception("stopping the motion source failed")
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception:  # noqa: BLE001
                logger.exception("closing the recorder failed")
        log_event(logger, logging.INFO, "stopped", dict(self.gate.counts))
