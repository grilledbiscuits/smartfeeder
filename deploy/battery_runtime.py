#!/usr/bin/env python3
"""Fsynced Pi runtime checkpoints for a battery discharge test.

The last complete row is a lower bound on runtime after sudden power loss.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    out_dir = Path(sys.argv[1])
    interval = float(os.environ.get("BATTERY_RUNTIME_INTERVAL", "2"))
    if interval <= 0:
        raise ValueError("BATTERY_RUNTIME_INTERVAL must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    path = out_dir / f"run-{datetime.now():%Y%m%dT%H%M%S}-{boot_id[:8]}-{os.getpid()}.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    dir_fd = os.open(out_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    start = time.monotonic_ns()

    def checkpoint(event: str) -> None:
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
            )
            throttled = result.stdout.strip().split("=", 1)[-1] if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            throttled = None
        row = {
            "event": event,
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_s": round((time.monotonic_ns() - start) / 1e9, 3),
            "boot_id": boot_id,
            "throttled": throttled,
        }
        line = json.dumps(row, separators=(",", ":")) + "\n"
        file.write(line)
        file.flush()
        os.fsync(file.fileno())

    with os.fdopen(fd, "w") as file:
        checkpoint("start")
        while not stopped.wait(interval):
            checkpoint("sample")
        checkpoint("stopped")


if __name__ == "__main__":
    main()
