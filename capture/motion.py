"""Motion sources: the HC-SR501 on GPIO, and a mock for off-Pi testing.

Event-driven, never polled. `gpiozero.MotionSensor` runs its own edge-detection
thread and calls `when_motion` on a rising edge, so this service spends its idle
time blocked rather than spinning -- which matters on a board that is also
expected to encode video.

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
    """VL53L1X Time-of-Flight sensor on I2C.

    The sensor measures distance continuously. When an object enters the
    configured detection range, motion is triggered. This avoids the false
    triggers and settling delays of passive infrared.

    Wiring (Raspberry Pi 4B):
    - I2C1 SDA (data): GPIO 2 (physical pin 3)
    - I2C1 SCL (clock): GPIO 3 (physical pin 5)
    - INT (optional interrupt): GPIO 17 (physical pin 11)
    - GND: any ground pin
    - VCC: any 3.3V pin

    If no interrupt pin is configured, the service polls the sensor at
    read_rate_hz. Polling is simpler and sufficient; interrupt-driven
    requires precise threshold tuning per installation.
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        i2c_address: int = 0x29,
        detection_range_mm: int = 500,
        warmup_seconds: float = 1.0,
        read_rate_hz: float = 10.0,
        interrupt_pin: int | None = None,
    ) -> None:
        self.i2c_bus = int(i2c_bus)
        self.i2c_address = int(i2c_address)
        self.detection_range_mm = int(detection_range_mm)
        self.warmup_seconds = float(warmup_seconds)
        self.read_rate_hz = float(read_rate_hz)
        self.interrupt_pin = int(interrupt_pin) if interrupt_pin else None
        self._sensor = None
        self._thread = None
        self._stop = threading.Event()
        self._callback: Callback | None = None
        self._last_motion_state = False

    def start(self, callback: Callback) -> None:
        try:
            from adafruit_circuitpython_vl53l1x import VL53L1X
            import board
            import busio
        except ImportError as exc:
            raise HardwareUnavailable(
                "adafruit-circuitpython-vl53l1x is not installed. Install with:\n"
                "  pip install adafruit-circuitpython-vl53l1x\n"
                "Also requires: sudo apt install python3-smbus2 python3-rpi-gpio"
            ) from exc

        try:
            # Initialize I2C on the specified bus
            if self.i2c_bus == 1:
                i2c = busio.I2C(board.SCL_1, board.SDA_1)
            elif self.i2c_bus == 0:
                i2c = busio.I2C(board.SCL, board.SDA)
            else:
                raise HardwareUnavailable(f"I2C bus {self.i2c_bus} not supported")

            self._sensor = VL53L1X(i2c, address=self.i2c_address)
        except Exception as exc:
            raise HardwareUnavailable(
                f"could not initialize VL53L1X on I2C{self.i2c_bus} at 0x{self.i2c_address:02x}: "
                f"{type(exc).__name__}: {exc}. Check wiring and I2C is enabled "
                "(raspi-config nonint do_i2c 0)."
            ) from exc

        self._callback = callback

        # Warmup period: sensor needs time to stabilize after power-on
        if self.warmup_seconds > 0:
            logger.info(
                "ToF on I2C%d (0x%02x): waiting %.0fs for sensor to stabilize",
                self.i2c_bus,
                self.i2c_address,
                self.warmup_seconds,
            )
            threading.Event().wait(self.warmup_seconds)

        logger.info(
            "ToF armed on I2C%d (0x%02x): %d mm detection range, polling at %.1f Hz",
            self.i2c_bus,
            self.i2c_address,
            self.detection_range_mm,
            self.read_rate_hz,
        )

        # Start polling thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="tof-poller", daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        """Poll sensor at configured rate and fire motion events on range changes."""
        import time

        interval = 1.0 / self.read_rate_hz
        while not self._stop.wait(interval):
            try:
                distance = self._sensor.distance
                motion_detected = distance < self.detection_range_mm

                if motion_detected and not self._last_motion_state:
                    self._last_motion_state = True
                    if self._callback:
                        self._callback(MotionEvent.now(Trigger.TOF))
                elif not motion_detected and self._last_motion_state:
                    self._last_motion_state = False

            except Exception as exc:
                logger.error("ToF polling error: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sensor is not None:
            self._sensor = None
            logger.info("ToF released on I2C%d", self.i2c_bus)
