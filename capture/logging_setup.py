"""Rotating, leveled, optionally-structured logging.

The service runs unattended for weeks, so the log is the only account of what
happened. Two formats:

* `text` -- readable, matches the format the ml/ modules use in their `main()`.
* `json` -- one JSON object per line, with the per-event fields inlined so a
  capture decision can be grepped or fed to jq without parsing prose.

Structured fields ride along in `extra={"fields": {...}}`. Under the text
formatter they are appended as `key=value` pairs; under JSON they become
top-level keys. Either way the call site is the same, so no module has to know
which format is configured.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

# Keys the JSON formatter owns. A structured field with one of these names is
# prefixed rather than dropped: a decision's taxonomic `level` silently
# overwriting the log's severity is the kind of bug that only shows up when you
# are grepping the log during an incident.
_RESERVED = {"ts", "level", "logger", "msg", "exc"}


class TextFormatter(logging.Formatter):
    """Human format with structured fields appended."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields:
            extras = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
            if extras:
                return f"{base} | {extras}"
        return base


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                payload[f"field_{key}" if key in _RESERVED else key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    level: str = "INFO",
    file: str | Path | None = None,
    *,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
    fmt: str = "text",
    console: bool = True,
) -> None:
    """Configure the root logger. Safe to call twice (handlers are replaced)."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    def make_formatter() -> logging.Formatter:
        if fmt == "json":
            return JsonFormatter()
        return TextFormatter("%(asctime)s %(levelname)-7s %(name)-24s %(message)s")

    if file:
        path = Path(file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=int(max_bytes), backupCount=int(backup_count), encoding="utf-8"
        )
        rotating.setFormatter(make_formatter())
        root.addHandler(rotating)

    if console:
        # stdout, not stderr: under systemd both land in the journal, and
        # keeping stderr clear means a genuine crash traceback stands out.
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(make_formatter())
        root.addHandler(stream)

    # picamera2 and gpiozero are chatty at DEBUG and their internals are not
    # this service's business.
    for noisy in ("picamera2", "libcamera", "gpiozero", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, level: int, msg: str, fields: dict[str, Any]) -> None:
    """Emit one structured line. Thin, but it keeps `extra=` spelling in one place."""
    logger.log(level, msg, extra={"fields": fields})
