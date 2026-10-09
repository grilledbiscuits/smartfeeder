"""Recording a clip: Camera Module 3 via picamera2, and a replay stand-in.

Camera Module 3 is an IMX708 behind libcamera, which the legacy `picamera`
library cannot drive at all -- `picamera2` is not a preference here, it is the
only option.

Encoding choices, and why
-------------------------
H.264 at 1280x720. `reports/deployment.md` records that the Pi 4B **keeps** the
hardware H.264 encoder that the Pi 5's BCM2712 dropped, so on this board the
encode is nearly free and does not compete with classification for CPU. 720p is
comfortably above what the classifier consumes -- it samples frames and
centre-crops to 224px -- so pushing to 1080p would cost bitrate and thermal
headroom to feed pixels that get thrown away. Every value is configurable; these
are the defaults, not assumptions baked into code.

Finalisation
------------
`stop_recording()` returning is not the same as the file being complete: the
mp4 muxer still has to write its moov atom, and the bytes still have to reach
the card. `_finalise` waits for a non-zero, stable size and then fsyncs, so the
clip handed to the classifier is a whole clip.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from capture.spool import fsync_dir, fsync_file

logger = logging.getLogger(__name__)


class CameraUnavailable(RuntimeError):
    """The camera could not be opened or a recording could not be written."""


@dataclass
class RecordingResult:
    path: Path
    duration_seconds: float
    size_bytes: int


class Recorder(Protocol):
    def record(self, dest: Path, seconds: float) -> RecordingResult: ...

    def close(self) -> None: ...


class FramePeeker(Protocol):
    """Grab a few frames WITHOUT starting an encode.

    This is what lets the empty-feeder gate run before the recorder rather than
    after it. Most motion events at a feeder are wind on the shade cloth, and on
    a Pi 5 -- which has no hardware H.264 encoder -- encoding each of those and
    then deleting it is the largest avoidable load on the board.

    A recorder that cannot peek simply does not implement this; the pipeline
    checks and skips the gate, which is the fail-open direction.
    """

    def peek(self, count: int = 3, interval_seconds: float = 0.25) -> list: ...


def _finalise(path: Path, timeout: float = 5.0) -> int:
    """Wait for the file to stop growing, then flush it to the card."""
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        if not path.is_file():
            time.sleep(0.05)
            continue
        size = path.stat().st_size
        if size > 0 and size == last:
            break
        last = size
        time.sleep(0.1)

    if not path.is_file() or path.stat().st_size == 0:
        raise CameraUnavailable(
            f"{path.name} is missing or empty after recording stopped. The "
            "encoder or the muxer failed; check dmesg for camera errors and the "
            "log for ffmpeg output."
        )
    fsync_file(path)
    fsync_dir(path.parent)
    return path.stat().st_size


class Picamera2Recorder:
    """Camera Module 3 via picamera2, held open between clips.

    Opening the camera costs roughly a second, which is a second of a bird's
    visit, so the device is opened once and reused. Any failure closes it so the
    next event starts from a clean state rather than inheriting a wedged one.
    """

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int,
        bitrate_kbps: int,
        *,
        warmup_seconds: float = 1.0,
        lens_position: float | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.framerate = int(framerate)
        self.bitrate = int(bitrate_kbps) * 1000
        self.warmup_seconds = float(warmup_seconds)
        # Fixed focus, in dioptres (1 / metres). None leaves the sensor alone.
        #
        # Camera Module 3 is an AUTOFOCUS module, and picamera2 leaves imx708 in
        # AfMode.Manual at LensPosition 1.0 -- one metre -- unless told
        # otherwise. Nothing told it otherwise until 2026-10-09, so every clip
        # ever recorded was focused roughly nine times too far away: the
        # background was sharp and the bird at the port was a blur. A confirmed
        # Cape White-eye visit was still identified correctly from blurred
        # frames (zosterops_virens top class on 9 of 12, peak 0.894), but the
        # novelty gate rejected 10 of those 12 as unfamiliar, which is what a
        # blurred subject looks like to a model trained on sharp ones.
        #
        # Manual rather than continuous AF on purpose: the perch does not move,
        # while continuous AF would hunt against an empty port and leave the
        # first frames of a visit -- the ones the classifier samples -- mid-sweep.
        self.lens_position = None if lens_position is None else float(lens_position)
        self._picam = None
        self._encoder_cls = None
        self._output_cls = None

    def _apply_focus(self, picam) -> None:
        """Pin the lens, after start() -- controls are rejected before it.

        A sensor with no focus motor (Camera Module 2, the HQ cam) raises rather
        than silently ignoring this, so it is a warning and not a failure: a
        fixed-focus camera that records is better than a service that will not
        start. Anything else is re-raised by the caller as CameraUnavailable.
        """
        if self.lens_position is None:
            return
        try:
            # AfModeEnum.Manual is 0 in libcamera's control enum. Taken from the
            # package when it is importable, but NOT depended on: a missing
            # libcamera import must not be the reason focus goes unset, and that
            # is exactly what happens off-Pi, where the tests run.
            manual = 0
            try:
                from libcamera import controls

                manual = int(controls.AfModeEnum.Manual)
            except Exception:  # noqa: BLE001 - the literal above is the contract
                pass

            picam.set_controls({"AfMode": manual, "LensPosition": self.lens_position})
            logger.info(
                "camera focus pinned at LensPosition %.2f (about %.3f m)",
                self.lens_position,
                1.0 / self.lens_position if self.lens_position else float("inf"),
            )
        except Exception as exc:  # noqa: BLE001 - a lens-less sensor must still record
            logger.warning(
                "could not set camera.lens_position=%s (%s: %s); leaving focus as the "
                "driver default, which on Camera Module 3 is 1.0 dioptres = 1 m",
                self.lens_position,
                type(exc).__name__,
                exc,
            )

    def _open(self):
        if self._picam is not None:
            return self._picam
        try:
            from picamera2 import Picamera2
            from picamera2.encoders import H264Encoder
            from picamera2.outputs import FfmpegOutput
        except ImportError as exc:
            raise CameraUnavailable(
                "picamera2 is not importable. On Raspberry Pi OS Bookworm it is "
                "an apt package, not a pip one:\n"
                "  sudo apt install -y python3-picamera2\n"
                "and the virtualenv must be created with --system-site-packages. "
                "The legacy `picamera` library does not support Camera Module 3."
            ) from exc

        self._encoder_cls, self._output_cls = H264Encoder, FfmpegOutput
        try:
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (self.width, self.height)},
                controls={
                    "FrameDurationLimits": (
                        int(1_000_000 / self.framerate),
                        int(1_000_000 / self.framerate),
                    )
                },
            )
            picam.configure(config)
            picam.start()
            self._apply_focus(picam)
        except Exception as exc:
            raise CameraUnavailable(
                f"could not open the camera: {type(exc).__name__}: {exc}. It may "
                "be held by another process (libcamera allows one client), or "
                "the ribbon cable may be seated wrong -- `libcamera-hello --list-cameras` "
                "is the quickest check."
            ) from exc

        # The AGC and AWB need a moment to converge; recording immediately gives
        # a clip that starts over- or under-exposed, which is precisely the
        # frames the classifier would sample first.
        if self.warmup_seconds > 0:
            time.sleep(self.warmup_seconds)

        self._picam = picam
        logger.info(
            "camera open: %dx%d @ %dfps, %d kbps H.264",
            self.width,
            self.height,
            self.framerate,
            self.bitrate // 1000,
        )
        return picam

    def peek(self, count: int = 3, interval_seconds: float = 0.25) -> list:
        """Grab frames from the live stream without starting the encoder.

        `capture_array` pulls from the preview stream, so no H.264 is produced
        and nothing is written to the card -- which is the entire point of doing
        this before `record` rather than after it.

        Returns [] on any failure. The caller treats an empty list as "cannot
        tell", and the gate fails open from there.
        """
        import numpy as np

        try:
            picam = self._open()
            if not getattr(picam, "started", False):
                picam.start()
                time.sleep(self.warmup_seconds)
            frames = []
            for i in range(max(1, int(count))):
                if i:
                    time.sleep(max(0.0, interval_seconds))
                frames.append(np.asarray(picam.capture_array("main"), dtype=np.float32)[..., :3])
            return frames
        except Exception as exc:  # noqa: BLE001 - never lose a capture over a peek
            logger.warning("peek failed: %s: %s", type(exc).__name__, exc)
            return []

    def snapshot(self):
        """One frame for the empty gate's background, then stop the camera.

        Taken only while the PIR is quiet, every few minutes. The camera is
        stopped again afterwards so an idle feeder does not keep the sensor and
        ISP running between snapshots; `peek` and `record` restart it.
        """
        frames = self.peek(count=1)
        try:
            if self._picam is not None and getattr(self._picam, "started", False):
                self._picam.stop()
        except Exception as exc:  # noqa: BLE001 - the next open recovers
            logger.warning("stopping the camera after a snapshot failed: %s", exc)
            self.close()
        return frames[0] if frames else None

    def record(self, dest: Path, seconds: float) -> RecordingResult:
        picam = self._open()
        dest.parent.mkdir(parents=True, exist_ok=True)
        encoder = self._encoder_cls(bitrate=self.bitrate)
        output = self._output_cls(str(dest))

        started = time.monotonic()
        try:
            picam.start_recording(encoder, output)
            time.sleep(seconds)
        except Exception as exc:
            self.close()
            raise CameraUnavailable(
                f"recording failed for {dest.name}: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            try:
                picam.stop_recording()
            except Exception as exc:  # pragma: no cover - hardware path
                logger.error("stop_recording raised: %s: %s", type(exc).__name__, exc)
                self.close()

        elapsed = time.monotonic() - started
        size = _finalise(dest)
        return RecordingResult(path=dest, duration_seconds=round(elapsed, 2), size_bytes=size)

    def close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.close()
            except Exception as exc:  # pragma: no cover - hardware path
                logger.warning("closing camera raised %s: %s", type(exc).__name__, exc)
            self._picam = None
            logger.info("camera closed")


class ReplayRecorder:
    """Copies a pre-recorded clip instead of using a camera.

    This is what makes the full decision pipeline exercisable off-Pi: point it
    at real feeder footage and every stage after it -- frame sampling, the
    classifier, the keep/discard rule, publication -- runs exactly as it does on
    the device.
    """

    def __init__(self, source: Path, *, simulate_duration: bool = False) -> None:
        self.source = Path(source)
        if not self.source.is_file():
            raise CameraUnavailable(f"replay clip not found: {self.source}")
        self.simulate_duration = simulate_duration
        self.calls = 0

    def record(self, dest: Path, seconds: float) -> RecordingResult:
        self.calls += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.simulate_duration:
            time.sleep(seconds)
        shutil.copyfile(self.source, dest)
        size = _finalise(dest)
        logger.info("replayed %s -> %s (%d bytes)", self.source.name, dest.name, size)
        return RecordingResult(path=dest, duration_seconds=float(seconds), size_bytes=size)

    def peek(self, count: int = 3, interval_seconds: float = 0.25) -> list:
        """Decode the first frames of the replay clip.

        Uses the same ffmpeg path the classifier uses, so off-Pi runs exercise
        the gate on real footage rather than on synthetic arrays.
        """
        import subprocess
        import tempfile

        import numpy as np
        from PIL import Image

        fps = max(0.1, 1.0 / max(interval_seconds, 1e-3))
        with tempfile.TemporaryDirectory() as tmp:
            pattern = str(Path(tmp) / "peek_%03d.jpg")
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(self.source),
                    "-vf",
                    f"fps={fps}",
                    "-frames:v",
                    str(int(count)),
                    "-q:v",
                    "3",
                    "-y",
                    pattern,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                logger.warning("peek failed on %s: %s", self.source.name, proc.stderr[-200:])
                return []
            out = []
            for f in sorted(Path(tmp).glob("peek_*.jpg")):
                with Image.open(f) as im:
                    out.append(np.asarray(im.convert("RGB"), dtype=np.float32))
            return out

    def close(self) -> None:
        return None
