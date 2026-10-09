#!/usr/bin/env python3
"""Append one CSV row of Pi health per interval, for the deployment chapter.

Runs as its own systemd service so the record survives reboots and is
independent of the capture service's own log rotation.

WHAT THIS CANNOT DO: measure power. A Pi 4B has no onboard power sensor, so
watts are not obtainable in software at any sample rate. `core_volt` is the SoC
core rail, not input power, and throttle flags only say whether the 5 V input
sagged below the undervoltage threshold. Real consumption needs an inline USB-C
power meter. Do not present anything here as a power measurement.

Columns:
  iso_time        local time, ISO-8601. The Pi has no RTC, so a row written
                  before NTP syncs carries a WRONG timestamp -- `synced` says
                  which, and rows with synced=0 should be dropped from any plot.
  uptime_s        seconds since boot, from the monotonic clock. Trustworthy
                  even when iso_time is not, so prefer it as the x-axis.
  temp_c          SoC temperature
  throttled       vcgencmd get_throttled bitfield, hex. 0x0 is clean.
                  bit0 undervolt now, bit1 arm freq capped, bit2 throttled now,
                  bit3 soft temp limit; bits 16-19 the same, latched since boot.
  arm_hz core_hz  measured clock frequencies
  core_volt       SoC core voltage
  cpu_pct         whole-machine CPU use since the previous row
  load1           1-minute load average
  capture_rss_mib RSS of the capture service's main process, 0 if not running
  mem_avail_mib   MemAvailable
  disk_free_mib   free space on /
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/smartfeeder/var/telemetry/health.csv")
INTERVAL = float(os.environ.get("TELEMETRY_INTERVAL", "30"))
COLUMNS = [
    "iso_time",
    "uptime_s",
    "synced",
    "temp_c",
    "throttled",
    "arm_hz",
    "core_hz",
    "core_volt",
    "cpu_pct",
    "load1",
    "capture_rss_mib",
    "mem_avail_mib",
    "disk_free_mib",
]


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def vcgen(*args: str) -> str:
    out = sh("vcgencmd", *args)
    return out.split("=", 1)[1] if "=" in out else ""


def cpu_times() -> tuple[float, float]:
    """(busy, total) jiffies from /proc/stat."""
    f = Path("/proc/stat").read_text().split("\n")[0].split()[1:]
    v = [float(x) for x in f]
    idle = v[3] + (v[4] if len(v) > 4 else 0.0)
    return sum(v) - idle, sum(v)


def capture_rss_mib() -> float:
    pid = sh("systemctl", "show", "-p", "MainPID", "--value", "birdcam-capture")
    if not pid.isdigit() or pid == "0":
        return 0.0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def meminfo_mib(key: str) -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key):
            return int(line.split()[1]) / 1024
    return 0.0


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists() or OUT.stat().st_size == 0:
        OUT.write_text(",".join(COLUMNS) + "\n")
    prev = cpu_times()
    while True:
        time.sleep(INTERVAL)
        now = cpu_times()
        dbusy, dtotal = now[0] - prev[0], now[1] - prev[1]
        prev = now
        st = os.statvfs("/")
        row = [
            datetime.now().isoformat(timespec="seconds"),
            f"{float(Path('/proc/uptime').read_text().split()[0]):.0f}",
            "1" if "synchronized: yes" in sh("timedatectl", "status") else "0",
            vcgen("measure_temp").removesuffix("'C"),
            sh("vcgencmd", "get_throttled").split("=", 1)[-1],
            vcgen("measure_clock", "arm"),
            vcgen("measure_clock", "core"),
            vcgen("measure_volts", "core").removesuffix("V"),
            f"{(100.0 * dbusy / dtotal if dtotal > 0 else 0.0):.1f}",
            open("/proc/loadavg").read().split()[0],
            f"{capture_rss_mib():.1f}",
            f"{meminfo_mib('MemAvailable'):.0f}",
            f"{st.f_bavail * st.f_frsize / 1048576:.0f}",
        ]
        with OUT.open("a") as fh:
            fh.write(",".join(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


if __name__ == "__main__":
    main()
