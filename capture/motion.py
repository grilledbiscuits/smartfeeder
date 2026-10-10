"""Motion sources: the HC-SR501 on GPIO, the VL53L1X ToF sensor on I2C, and a
mock for off-Pi testing.

The PIR is event-driven, never polled. `gpiozero.MotionSensor` runs its own edge-detection
thread and calls `when_motion` on a rising edge, so this service spends its idle
time blocked rather than spinning -- which matters on a board that is also
expected to encode video. The ToF sensor is polled; see `ToFMotionSource`.

The HC-SR501 has hardware retrigger behaviour of its own (the on-board Tx
potentiometer holds the output high for a tunable period after motion), so
`when_motion` can fire again the moment that period lapses while the same bird
is still at the feeder. The software cooldown in `service.TriggerGate` is what
turns that into one event; nothing here tries to second-guess the sensor.
"""

from __future__ import annotations

import logging
import statistics
import threading
from collections import deque
from collections.abc import Callable
from typing import Protocol

from capture.events import MotionEvent, Trigger

logger = logging.getLogger(__name__)

Callback = Callable[[MotionEvent], None]


class HardwareUnavailable(RuntimeError):
    """GPIO could not be claimed. Carries the operator-facing explanation."""


class MotionSource(Protocol):
    """Anything that can announce motion."""

    def start(self, callback: Callback) -> None: ...

    def stop(self) -> None: ...


class PirMotionSource:
    """HC-SR501 on a GPIO input via gpiozero."""

    def __init__(
        self,
        pin: int,
        *,
        sample_rate_hz: float = 10.0,
        queue_len: int = 1,
        warmup_seconds: float = 60.0,
    ) -> None:
        self.pin = int(pin)
        self.sample_rate_hz = float(sample_rate_hz)
        # queue_len=1 means "report the raw pin state, no averaging". The
        # HC-SR501 already debounces in hardware and holds its output high for
        # its whole retrigger window; averaging on top only delays the first
        # edge, which is the one that matters for catching a bird landing.
        self.queue_len = int(queue_len)
        self.warmup_seconds = float(warmup_seconds)
        self._sensor = None

    def start(self, callback: Callback) -> None:
        try:
            from gpiozero import MotionSensor
        except ImportError as exc:  # pragma: no cover - hardware path
            raise HardwareUnavailable(
                "gpiozero is not installed. On Raspberry Pi OS Bookworm:\n"
                "  sudo apt install python3-gpiozero python3-lgpio\n"
                "and create the venv with --system-site-packages."
            ) from exc

        try:
            self._sensor = MotionSensor(
                self.pin,
                sample_rate=self.sample_rate_hz,
                queue_len=self.queue_len,
            )
        except Exception as exc:  # pragma: no cover - hardware path
            # gpiozero raises several distinct types here (GPIOPinInUse,
            # BadPinFactory, PinInvalidPin). They all mean the same thing to an
            # operator, and the fix is the same.
            raise HardwareUnavailable(
                f"could not claim GPIO{self.pin} for the PIR sensor: "
                f"{type(exc).__name__}: {exc}. Another process may hold the pin "
                "(check for a second copy of this service), or the user may not "
                "be in the 'gpio' group."
            ) from exc

        # The HC-SR501 emits spurious highs while its pyroelectric sensor
        # settles. Warming up before wiring the callback avoids a burst of
        # recordings every time the service restarts.
        if self.warmup_seconds > 0:
            logger.info(
                "PIR on GPIO%d: waiting %.0fs for the sensor to settle before arming",
                self.pin,
                self.warmup_seconds,
            )
            self._sensor.wait_for_no_motion(timeout=self.warmup_seconds)

        self._sensor.when_motion = lambda: callback(MotionEvent.now(Trigger.PIR))
        logger.info("PIR armed on GPIO%d", self.pin)

    def stop(self) -> None:
        if self._sensor is not None:
            self._sensor.when_motion = None
            self._sensor.close()
            self._sensor = None
            logger.info("PIR on GPIO%d released", self.pin)


class MockMotionSource:
    """A PIR you can fire by hand, for testing the pipeline off-Pi.

    Either call `trigger()` directly from a test, or hand it a schedule and let
    it fire on a background thread the way the real sensor does.
    """

    def __init__(self, schedule: list[float] | None = None) -> None:
        self.schedule = list(schedule or [])
        self._callback: Callback | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.fired = 0

    def start(self, callback: Callback) -> None:
        self._callback = callback
        logger.info("mock motion source armed (%d scheduled trigger(s))", len(self.schedule))
        if self.schedule:
            self._thread = threading.Thread(target=self._run, name="mock-pir", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        for delay in self.schedule:
            if self._stop.wait(delay):
                return
            self.trigger()

    def trigger(self) -> MotionEvent:
        """Fire one motion event synchronously. Returns what was delivered."""
        event = MotionEvent.now(Trigger.MOCK)
        self.fired += 1
        if self._callback is not None:
            self._callback(event)
        return event

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class ToFMotionSource:
    """VL53L1X Time-of-Flight sensor on I2C, polled.

    The sensor ranges continuously and the source fires on a *change* in what
    it sees, not on an absolute distance. Unlike passive infrared, a bird
    sitting still at the port stays detected without re-triggering.

    Two detection models, chosen by `baseline_margin_mm`:

    - **baseline-relative (set it; this is the deployed default).** The source
      learns the resting distance of the empty scene and fires when a reading
      departs from it by at least the margin. This is the only model that works
      at this feeder, because the perch is permanently in the beam -- it is
      bolted there, by design. See below.
    - **absolute (leave it null).** Fires on anything inside
      `detection_range_mm`. Kept for a mounting where the beam really is empty
      at rest, and for the replay tests.

    Why absolute cannot work here. The question `detection_range_mm` asks is
    "is anything within 50 cm?", and the perch answers yes forever. So the gate
    latched on the first poll after start and never produced another rising
    edge: birds landed on an already-triggered sensor and generated no event.
    That is what made 2026-10-09 look like a threshold problem for an
    afternoon. Narrowing `roi_size` appeared to fix it only by aiming the beam
    away from the place birds land, and the `max_hold_seconds` valve -- added
    to log the stuck gate -- became the only thing admitting any detection at
    all, on a 60 s timer unrelated to the birds.

    This is the same model the camera's `IdleBackgroundGate` already uses:
    learn what the empty scene looks like, trigger on departure from it.

    Wiring (Raspberry Pi 4B):
    - SDA: GPIO 2 (physical pin 3)
    - SCL: GPIO 3 (physical pin 5)
    - VIN: 3.3 V, GND: any ground

    Polled, not interrupt-driven: the Pi 4B has no sleep state, so an interrupt
    line saves no measurable power, and polling needs no threshold registers
    programmed into the sensor. `interrupt_pin` is accepted for config
    compatibility and ignored.

    Driver notes (adafruit_vl53l1x 1.2.x): `distance` is in CENTIMETRES and is
    None when the reading is invalid (nothing in range, or ambient-light
    saturation); a new reading is only produced after `clear_interrupt()`.
    """

    # Short mode ranges to ~1.3 m and copes better with sunlight than long
    # mode, which only matters beyond 1.3 m -- far past the feeder port.
    SHORT_DISTANCE_MODE = 1

    def __init__(
        self,
        i2c_bus: int = 1,
        i2c_address: int = 0x29,
        detection_range_mm: int = 500,
        warmup_seconds: float = 1.0,
        read_rate_hz: float = 10.0,
        interrupt_pin: int | None = None,
        release_seconds: float = 1.0,
        max_hold_seconds: float | None = 60.0,
        roi_size: int | None = None,
        roi_center: int | None = None,
        baseline_margin_mm: int | None = None,
        baseline_samples: int = 1200,
        *,
        sensor_factory: Callable[[], object] | None = None,
    ) -> None:
        self.i2c_bus = int(i2c_bus)
        self.i2c_address = int(i2c_address)
        self.detection_range_mm = int(detection_range_mm)
        # Departure from the resting distance that counts as a detection, in mm.
        # None selects the absolute model instead (see the class docstring).
        #
        # Symmetric on purpose: |reading - baseline| >= margin.
        #
        # MEASURED 2026-10-10, and it is the opposite of what was assumed here
        # first: a bird at the port reads FARTHER than the resting baseline, not
        # nearer. Two confirmed visits logged +13 mm (Cape White-eye) and +11 mm
        # (Cape Bulbul) against a 118 mm baseline, matching the +11 mm of the
        # 2026-10-09 hand test. The resting return is the near furniture; a body
        # at the port scatters the beam so the dominant return comes from behind
        # it. A NEARER-ONLY margin would have missed both birds, so do not add
        # one -- the earlier note claiming birds read nearer was a guess, and
        # acting on it would have been the most expensive kind of wrong.
        #
        # FITTED 10 mm, 2026-10-10: 180 s at the feeder in SE wind, replayed
        # through poll_once. 0.3 false triggers/min against 5.2 at 7 mm, and
        # nothing above 10 mm improves on it. The earlier 7 mm and 20 mm fits
        # came from a replay harness that omitted the max_hold valve.
        #
        # The UPPER bound is unmeasured and is the live risk: the one measured
        # object-at-the-perch signal is ~11 mm and the wind tail reaches +11 mm,
        # so a sunbird may depart by less than the margin. Every rising edge
        # logs its departure now; fit the upper bound from those against
        # observer-confirmed visits before trusting this number.
        self.baseline_margin_mm = None if baseline_margin_mm is None else int(baseline_margin_mm)
        # Readings held for the running median, ALL of them, not just the ones
        # taken while the gate is clear. At 10 Hz, 1200 samples is a 120 s
        # window.
        #
        # MEASURED 2026-10-10, and this is the whole case for the long window.
        # The true resting distance, from a 180 s raw trace, is a median of
        # 118 mm. A 30-sample (3 s) window learned 126 mm, 8 mm off and most of
        # a 10 mm margin; 1200 samples learned 117 mm.
        #
        # Detected readings are no longer excluded either. Clear-state-only
        # updating is self-reinforcing in principle -- a baseline that drifts
        # keeps the gate detected, and excluded readings cannot pull it back --
        # which is a property of the code with its own test, NOT a failure
        # observed here. The 126->131 mm wander in that day's log is a 3 s median
        # jittering against a 3.0 mm-stdev scene, and was written up as a
        # ratchet in error.
        #
        # The first learn is the exposed case regardless: there is no baseline
        # yet, so nothing can be excluded as "detected" and whatever sits at the
        # port defines "empty". Window length is the only protection there, and
        # a visit is 7-25% of a 120 s median.
        #
        # Cost: the gate does not fire until the window is full, so there is a
        # ~120 s blind period after every restart. That is the trade for not
        # learning a bird as furniture, and a restart is rare; if restarts become
        # frequent, persist the baseline across them rather than shortening this.
        self.baseline_samples = max(1, int(baseline_samples))
        self._baseline_window: deque[float] = deque(maxlen=self.baseline_samples)
        self._baseline: float | None = None
        self._baseline_logged = False
        self.warmup_seconds = float(warmup_seconds)
        self.read_rate_hz = float(read_rate_hz)
        self.release_seconds = float(release_seconds)
        # Region of interest on the sensor's 16x16 SPAD array: `roi_size` is the
        # window's side in SPADs (4-16, default 16 = the full ~27 degree cone) and
        # `roi_center` is the centre SPAD, 0-255.
        #
        # This is how the feeder bottle is kept out of the beam. MEASURED
        # 2026-10-06 with the sensor mounted to the side of the neck: the bottle
        # reads a rock-steady 13 cm, which is NEARER than the perch at ~20 cm, and
        # the sensor reports one dominant return per reading -- so with the full
        # array a bird at the perch is masked behind the bottle and nothing can
        # recover it downstream. Narrowing to 4x4 drops the bottle into the
        # 239/247 corner of the array and leaves the centre clear.
        #
        # Narrower means less signal: a 4x4 window is about 7 degrees and returns
        # fewer valid reads off a small target, which is what release_seconds is
        # there to absorb. Measure before changing it; ROI readings taken without
        # restarting ranging are not reproducible (see _apply_roi).
        self.roi_size = None if roi_size is None else int(roi_size)
        self.roi_center = None if roi_center is None else int(roi_center)
        if self.roi_size is not None and not 4 <= self.roi_size <= 16:
            raise ValueError(f"motion.roi_size must be between 4 and 16, got {self.roi_size}")
        if self.roi_center is not None and not 0 <= self.roi_center <= 255:
            raise ValueError(f"motion.roi_center must be between 0 and 255, got {self.roi_center}")
        # Consecutive non-detections needed to re-arm. See poll_once: the sensor
        # drops roughly half its reads at the feeder, and an invalid read is
        # indistinguishable from an empty port, so re-arming on one of them turns
        # a single visit into an event per surviving read.
        self._release_polls = max(1, round(self.release_seconds * self.read_rate_hz))
        # Safety valve. A target that never clears holds the trigger down forever
        # and the feeder stops recording with nothing in the log to say so -- the
        # failure mode is a silent one, which is why it was misdiagnosed twice on
        # 2026-10-06. Anything in range for this long is treated as a stuck gate:
        # say so once, release, and let the next poll re-fire. A bird that really
        # does sit this long simply gets a second event, which the TriggerGate
        # cooldown already rate-limits. None disables the valve.
        self.max_hold_seconds = None if max_hold_seconds is None else float(max_hold_seconds)
        self._max_hold_polls = (
            None
            if not self.max_hold_seconds
            else max(1, round(self.max_hold_seconds * self.read_rate_hz))
        )
        if interrupt_pin is not None:
            logger.warning("motion.interrupt_pin is set but ignored: the ToF sensor is polled")
        # Injected in tests; the default opens the real sensor.
        self._sensor_factory = sensor_factory or self._open_sensor
        self._sensor = None
        self._thread = None
        self._stop = threading.Event()
        self._callback: Callback | None = None
        self._in_range = False
        self._clear_run = 0
        self._hold_run = 0

    def _open_sensor(self):  # pragma: no cover - hardware path
        try:
            import adafruit_vl53l1x
            import board
        except ImportError as exc:
            raise HardwareUnavailable(
                "the VL53L1X driver is not installed. In the service venv:\n"
                "  pip install adafruit-circuitpython-vl53l1x\n"
                "and enable I2C: sudo raspi-config nonint do_i2c 0"
            ) from exc
        if self.i2c_bus != 1:
            raise HardwareUnavailable(
                f"I2C bus {self.i2c_bus} is not supported; the sensor must be on I2C1 (GPIO 2/3)"
            )
        try:
            return adafruit_vl53l1x.VL53L1X(board.I2C(), address=self.i2c_address)
        except Exception as exc:
            raise HardwareUnavailable(
                f"could not initialise the VL53L1X on I2C{self.i2c_bus} "
                f"at 0x{self.i2c_address:02x}: "
                f"{type(exc).__name__}: {exc}. Check the wiring, that I2C is enabled "
                "(sudo raspi-config nonint do_i2c 0) and that `i2cdetect -y 1` lists the address."
            ) from exc

    def start(self, callback: Callback) -> None:
        self._sensor = self._sensor_factory()
        self._sensor.distance_mode = self.SHORT_DISTANCE_MODE
        self._apply_roi()
        self._sensor.start_ranging()
        self._callback = callback

        # Let the first readings settle before any of them can fire an event.
        if self.warmup_seconds > 0:
            logger.info("ToF: waiting %.1fs for the sensor to settle", self.warmup_seconds)
            self._stop.wait(self.warmup_seconds)

        logger.info(
            "ToF armed on I2C%d (0x%02x): %s, polling at %.1f Hz",
            self.i2c_bus,
            self.i2c_address,
            (
                f"{self.detection_range_mm} mm absolute detection range"
                if self.baseline_margin_mm is None
                else f"baseline-relative, +/-{self.baseline_margin_mm} mm from the "
                f"resting distance (learning from {self.baseline_samples} reads)"
            ),
            self.read_rate_hz,
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="tof-poller", daemon=True)
        self._thread.start()

    def _apply_roi(self) -> None:
        """Set the SPAD window, BEFORE ranging starts.

        Order matters. Changing the ROI while ranging is already running gives
        readings that do not reproduce -- measured 2026-10-06, the same centre on
        the same static scene returned "clear" and "13 cm" in alternating runs,
        which sent this investigation down a blind alley. Set it once, then start.

        A sensor stub without these attributes is fine: the tests use one.
        """
        if self.roi_size is None and self.roi_center is None:
            return
        try:
            if self.roi_size is not None:
                self._sensor.roi_xy = (self.roi_size, self.roi_size)
            if self.roi_center is not None:
                self._sensor.roi_center = self.roi_center
            logger.info(
                "ToF region of interest: %s SPADs centred on %s",
                "default" if self.roi_size is None else f"{self.roi_size}x{self.roi_size}",
                "default" if self.roi_center is None else self.roi_center,
            )
        except AttributeError:
            logger.warning(
                "this VL53L1X driver exposes no ROI control; motion.roi_size/roi_center "
                "are being ignored and the full 16x16 array is in use"
            )

    def _detect(self, distance_mm: float | None) -> bool:
        """Is this reading a detection? Also maintains the baseline.

        An invalid read is never a detection: the VL53L1X returns None both for
        "nothing in range" and for a dropout, and about half the reads at this
        feeder are dropouts. `release_seconds` is what distinguishes the two.

        Baseline learning consumes every valid in-scene reading. What stops a
        bird teaching the sensor that it is furniture is the LENGTH of the
        window, not a filter on which readings enter it: a visit is a small
        minority of a 120 s median. Filtering by gate state instead was worse
        than useless -- it let a wrong baseline reinforce itself.
        """
        if distance_mm is None:
            return False
        if self.baseline_margin_mm is None:
            return distance_mm < self.detection_range_mm
        # The absolute range still applies as an outer bound: a return from the
        # garden beyond it is not the feeder scene and must not move the
        # baseline, however far it departs from it.
        if distance_mm >= self.detection_range_mm:
            return False
        # Every valid in-scene reading feeds the median, including ones taken
        # while the gate is detected. See the note on baseline_samples: excluding
        # them is what let the baseline ratchet away from the resting distance.
        self._observe_baseline(distance_mm)
        if self._baseline is None:
            return False  # still filling the window; nothing to compare against
        return abs(distance_mm - self._baseline) >= self.baseline_margin_mm

    def _observe_baseline(self, distance_mm: float) -> None:
        """Fold one reading into the running median, and keep it current."""
        self._baseline_window.append(distance_mm)
        if len(self._baseline_window) < self.baseline_samples:
            return
        self._baseline = statistics.median(self._baseline_window)
        if not self._baseline_logged:
            self._baseline_logged = True
            logger.info(
                "ToF resting baseline learned: %.0f mm (%.1f cm) from %d reads; "
                "firing on a departure of %d mm or more",
                self._baseline,
                self._baseline / 10.0,
                len(self._baseline_window),
                self.baseline_margin_mm,
            )

    def poll_once(self) -> None:
        """Read one sample, if ready, and fire on the clear -> detected edge.

        Detection arms immediately; releasing needs `release_seconds` of unbroken
        non-detection. That asymmetry is the point. MEASURED at the feeder
        2026-10-06: over 610 polls with a hand moving at the port, only 50% of
        reads returned a distance at all -- the rest came back None. Because the
        VL53L1X reports "nothing in range" as None too, an invalid read cannot be
        told apart from an empty port, so releasing on the first one turned one
        visit into 44 separate triggers in 60 seconds. Ignoring None instead is
        not an option: the trigger would latch on and never re-arm.

        `capture.service.TriggerGate`'s cooldown does not cover this. It
        suppresses re-triggers for 30 s AFTER an admitted event, which is why the
        thrashing showed up as one admitted clip plus a long tail of
        `dropped_cooldown`; the sub-second flicker underneath it is this method's
        problem to absorb.
        """
        if not self._sensor.data_ready:
            return
        distance_cm = self._sensor.distance
        self._sensor.clear_interrupt()
        distance_mm = None if distance_cm is None else distance_cm * 10.0
        detected = self._detect(distance_mm)

        if detected:
            self._clear_run = 0
            if not self._in_range:
                self._in_range = True
                self._hold_run = 0
                # Log the departure that fired, not just that something did.
                # Without this a soak cannot tell a bird from a gust after the
                # fact, so the next margin decision would be another guess: the
                # empty-scene tail and the one measured object-at-the-perch
                # signal are both about 11 mm, and which side of that a real
                # sunbird falls on is the open question. One line per rising
                # edge, which the margin itself rate-limits.
                if self.baseline_margin_mm is not None and self._baseline is not None:
                    logger.info(
                        "ToF trigger: %.0f mm vs baseline %.0f mm (departure %+.0f mm, "
                        "margin %d mm)",
                        distance_mm,
                        self._baseline,
                        distance_mm - self._baseline,
                        self.baseline_margin_mm,
                    )
                if self._callback:
                    self._callback(MotionEvent.now(Trigger.TOF))
                return
            self._hold_run += 1
            if self._max_hold_polls is not None and self._hold_run >= self._max_hold_polls:
                logger.warning(
                    "ToF has read detected for %.0fs without clearing (last %.1f cm, "
                    "baseline %s). Treating the gate as stuck and re-arming. In "
                    "baseline mode this means the learned resting distance no longer "
                    "matches the scene -- the mount has moved, or it was learned with "
                    "something at the port. The running median will now converge on "
                    "the new scene; before 2026-10-10 it could not, because detected "
                    "readings were excluded from it.",
                    self.max_hold_seconds,
                    distance_cm if distance_cm is not None else float("nan"),
                    "unlearned" if self._baseline is None else f"{self._baseline:.0f} mm",
                )
                self._in_range = False
                self._hold_run = 0
            return

        # Not detected: ride out a dropout, but let a departure through.
        self._hold_run = 0
        self._clear_run += 1
        if self._clear_run >= self._release_polls:
            self._in_range = False

    def _poll_loop(self) -> None:
        interval = 1.0 / self.read_rate_hz
        while not self._stop.wait(interval):
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - one bad I2C read must not kill the poller
                logger.error("ToF read failed: %s: %s", type(exc).__name__, exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sensor is not None:
            try:
                self._sensor.stop_ranging()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                logger.exception("stopping the ToF sensor failed")
            self._sensor = None
            logger.info("ToF released on I2C%d", self.i2c_bus)
