"""Configuration loading and validation for the capture service.

Follows the convention `birdcam.config` established: one YAML file is the
single source of truth, nothing is hardcoded in Python, and `validate()`
reports every problem at once and then raises. A capture service that starts
with a cooldown shorter than its clip length does not fail -- it quietly
records overlapping clips at 3am and fills the SD card, which is exactly the
class of mistake a fatal cross-check exists to prevent.

There are deliberately NO silent defaults. A missing key is an error naming
the key, not a value invented at import time. `config/capture.example.yaml`
is therefore the complete reference for every tunable.

Secrets
-------
Any string value may contain `${VAR}` or `${VAR:-fallback}` and is expanded
from the environment at load time. The same-host publisher needs no
credentials, but the mechanism is here so a token never has to be written to
disk when one is needed. An unset `${VAR}` with no fallback is a fatal error,
not an empty string -- silently authenticating as nobody is worse than not
starting.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class CaptureConfigError(RuntimeError):
    """Raised when the capture config is missing keys or internally inconsistent.

    Fatal by design; see the module docstring.
    """


def expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-fallback} in strings."""
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, fallback = m.group(1), m.group(2)
            env = os.environ.get(name)
            if env is not None:
                return env
            if fallback is not None:
                return fallback
            raise CaptureConfigError(
                f"Config references ${{{name}}} but that environment variable is "
                f"not set and no fallback was given. Either export {name} or "
                f"write ${{{name}:-default}}."
            )

        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


@dataclass
class CaptureConfig:
    """Loaded, cross-validated capture configuration."""

    raw: dict[str, Any]
    path: Path
    root: Path

    # -- loading ---------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, root: Path | None = None) -> CaptureConfig:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise CaptureConfigError(f"No capture config at {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise CaptureConfigError(f"{path} did not parse to a mapping.")
        obj = cls(
            raw=expand_env(data),
            path=path,
            # Relative paths resolve against the repo root, matching birdcam's
            # rule that no absolute path appears in any config file.
            root=Path(root) if root else Path(__file__).resolve().parent.parent,
        )
        obj.validate()
        return obj

    # -- access ----------------------------------------------------------------

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise CaptureConfigError(f"Config section {name!r} is missing from {self.path}")
        return value

    def get(self, dotted: str) -> Any:
        """Fetch `section.key`, raising a message that names the missing key."""
        section, _, key = dotted.partition(".")
        block = self.section(section)
        if key not in block:
            raise CaptureConfigError(f"Missing config key {dotted!r} in {self.path}")
        return block[key]

    def resolve_path(self, dotted: str) -> Path:
        """A configured path, resolved against the repo root if relative."""
        p = Path(str(self.get(dotted))).expanduser()
        return p if p.is_absolute() else (self.root / p)

    # -- validation ------------------------------------------------------------

    _SCHEMA: ClassVar[dict[str, tuple[str, ...]]] = {
        "gpio": ("pin", "sample_rate_hz", "queue_len", "warmup_seconds"),
        "camera": ("width", "height", "framerate", "bitrate_kbps", "warmup_seconds"),
        "capture": ("clip_seconds", "cooldown_seconds", "on_busy", "max_queued"),
        "storage": (
            "work_dir",
            "pending_dir",
            "review_dir",
            "min_free_mb",
            "max_pending_clips",
            "max_review_clips",
            "delete_after_publish",
        ),
        "classifier": (
            "enabled",
            "onnx_path",
            "sample_fps",
            "max_frames",
            "providers",
            "operating_points",
            "site_prior",
            "expect_source",
            "allow_artefact_mismatch",
            "keep_frames",
            "novelty",
        ),
        "publish": (
            "retain_uncertain",
            "escalate_after_attempts",
            "backoff_initial_seconds",
            "backoff_max_seconds",
            "backoff_factor",
        ),
        "logging": ("level", "file", "max_bytes", "backup_count", "format"),
    }

    def validate(self) -> None:
        """Cross-check the config. Collects every problem, then raises once.

        Each check below corresponds to a failure that would otherwise appear
        as a full SD card, an overlapping recording, or a service that runs for
        a week and records nothing.
        """
        problems: list[str] = []

        for section, keys in self._SCHEMA.items():
            block = self.raw.get(section)
            if not isinstance(block, dict):
                problems.append(f"section {section!r} is missing or not a mapping")
                continue
            for key in keys:
                if key not in block:
                    problems.append(f"missing key {section}.{key}")

        if problems:
            raise CaptureConfigError(
                f"Capture configuration {self.path} is incomplete:\n  - "
                + "\n  - ".join(problems)
                + "\n\nSee capture/config/capture.example.yaml for the full reference."
            )

        cap, storage, cam = self.section("capture"), self.section("storage"), self.section("camera")
        pub, cls_ = self.section("publish"), self.section("classifier")

        clip = float(cap["clip_seconds"])
        cooldown = float(cap["cooldown_seconds"])
        if clip <= 0:
            problems.append(f"capture.clip_seconds must be positive, got {clip}")
        if cooldown < clip:
            # The cooldown starts when recording STARTS, so a cooldown shorter
            # than the clip admits a second event mid-recording -- the exact
            # overlapping-recording bug the cooldown exists to prevent.
            problems.append(
                f"capture.cooldown_seconds ({cooldown}) is shorter than "
                f"capture.clip_seconds ({clip}). The cooldown starts at the "
                "beginning of a recording, so this would admit a second trigger "
                "while the first clip is still being written."
            )

        if cap["on_busy"] not in {"queue", "drop"}:
            problems.append(f"capture.on_busy must be 'queue' or 'drop', got {cap['on_busy']!r}")
        if int(cap["max_queued"]) < 0:
            problems.append("capture.max_queued must be >= 0")
        if cap["on_busy"] == "queue" and int(cap["max_queued"]) == 0:
            problems.append(
                "capture.on_busy is 'queue' but capture.max_queued is 0, which "
                "drops every concurrent trigger. Set on_busy: drop to say that "
                "explicitly, or raise max_queued."
            )

        for key in ("width", "height", "framerate", "bitrate_kbps"):
            if int(cam[key]) <= 0:
                problems.append(f"camera.{key} must be positive")
        if int(cam["width"]) % 2 or int(cam["height"]) % 2:
            # H.264 chroma subsampling requires even dimensions; the encoder
            # rejects odd ones with an error that does not name the config.
            problems.append("camera.width and camera.height must both be even (H.264 4:2:0)")

        if int(storage["min_free_mb"]) <= 0:
            problems.append("storage.min_free_mb must be positive; storage hygiene depends on it")
        for key in ("max_pending_clips", "max_review_clips"):
            if int(storage[key]) < 1:
                problems.append(f"storage.{key} must be at least 1")

        # A pending clip is one whose publication is unconfirmed, so it may
        # never be deleted to make room. The free-space floor must therefore be
        # able to hold the cap, or the service deadlocks: unable to record, and
        # unable to free anything.
        est_mb = clip * int(cam["bitrate_kbps"]) / 8 / 1024
        needed = est_mb * int(storage["max_pending_clips"])
        if needed > int(storage["min_free_mb"]) * 20:
            problems.append(
                f"storage.max_pending_clips ({storage['max_pending_clips']}) at "
                f"~{est_mb:.1f} MB per clip could hold {needed:.0f} MB of "
                f"undeletable clips against a {storage['min_free_mb']} MB free-space "
                "floor. Lower the cap or raise the floor."
            )

        if int(pub["escalate_after_attempts"]) < 1:
            problems.append("publish.escalate_after_attempts must be at least 1")
        if float(pub["backoff_factor"]) < 1.0:
            problems.append("publish.backoff_factor must be >= 1.0 or backoff shrinks")
        if float(pub["backoff_initial_seconds"]) <= 0:
            problems.append("publish.backoff_initial_seconds must be positive")
        if float(pub["backoff_max_seconds"]) < float(pub["backoff_initial_seconds"]):
            problems.append("publish.backoff_max_seconds is below backoff_initial_seconds")

        if cls_["enabled"]:
            if float(cls_["sample_fps"]) <= 0:
                problems.append("classifier.sample_fps must be positive")
            if int(cls_["max_frames"]) < 1:
                problems.append("classifier.max_frames must be at least 1")
            if not isinstance(cls_["providers"], list) or not cls_["providers"]:
                problems.append("classifier.providers must be a non-empty list")
            novelty = cls_["novelty"]
            if not isinstance(novelty, dict) or "enabled" not in novelty:
                problems.append("classifier.novelty must be a mapping with an 'enabled' key")

        if problems:
            raise CaptureConfigError(
                f"Capture configuration {self.path} is inconsistent:\n  - "
                + "\n  - ".join(problems)
            )

    # -- convenience -----------------------------------------------------------

    def summary(self) -> str:
        cam, cap = self.section("camera"), self.section("capture")
        cls_ = self.section("classifier")
        return "\n".join(
            [
                f"config          : {self.path}",
                f"repo root       : {self.root}",
                f"PIR pin         : GPIO{self.get('gpio.pin')}",
                f"clip            : {cap['clip_seconds']}s @ "
                f"{cam['width']}x{cam['height']}/{cam['framerate']}fps "
                f"{cam['bitrate_kbps']}kbps",
                f"cooldown        : {cap['cooldown_seconds']}s, "
                f"on_busy={cap['on_busy']} max_queued={cap['max_queued']}",
                f"classifier      : {'on' if cls_['enabled'] else 'OFF'} "
                f"({cls_['onnx_path']})",
                f"novelty gate    : {'on' if cls_['novelty'].get('enabled') else 'OFF'}",
                f"work dir        : {self.resolve_path('storage.work_dir')}",
                f"pending dir     : {self.resolve_path('storage.pending_dir')}",
                f"review dir      : {self.resolve_path('storage.review_dir')}",
                f"retain uncertain: {self.get('publish.retain_uncertain')}",
                f"delete on ok    : {self.get('storage.delete_after_publish')}",
            ]
        )
