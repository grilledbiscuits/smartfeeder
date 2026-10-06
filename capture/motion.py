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
import threading
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

    The sensor ranges continuously; a motion event fires on the first reading
    that comes inside `detection_range_mm`, and the source re-arms once a
    reading is out of range again. Unlike passive infrared, a bird sitting
    still at the port stays "in range" without re-triggering.

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
        *,
        sensor_factory: Callable[[], object] | None = None,
    ) -> None:
        self.i2c_bus = int(i2c_bus)
        self.i2c_address = int(i2c_address)
        self.detection_range_mm = int(detection_range_mm)
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
            "ToF armed on I2C%d (0x%02x): %d mm detection range, polling at %.1f Hz",
            self.i2c_bus,
            self.i2c_address,
            self.detection_range_mm,
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
        detected = distance_cm is not None and distance_cm * 10 < self.detection_range_mm

        if detected:
            self._clear_run = 0
            if not self._in_range:
                self._in_range = True
                self._hold_run = 0
                if self._callback:
                    self._callback(MotionEvent.now(Trigger.TOF))
                return
            self._hold_run += 1
            if self._max_hold_polls is not None and self._hold_run >= self._max_hold_polls:
                logger.warning(
                    "ToF has read in range for %.0fs without clearing (last %.1f cm). "
                    "Treating the gate as stuck and re-arming: check whether something "
                    "is parked in the beam, e.g. the perch itself.",
                    self.max_hold_seconds,
                    distance_cm if distance_cm is not None else float("nan"),
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
