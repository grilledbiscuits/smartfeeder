#!/usr/bin/env python3
"""Plot Pi health over the 2026-10-10 morning run for the report (Results, sec 5.3).

    .venv/bin/python deploy/telemetry-20261012/plot_telemetry.py OUT.pdf

Inputs sit beside this script: health.csv (copied from the Pi's
var/telemetry/health.csv on 2026-10-12) and events.txt (capture decided /
motion trigger dropped lines from capture.log, "date time outcome").
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
START, END = "2026-10-10 10:05", "2026-10-10 11:56"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

d = pd.read_csv(HERE / "health.csv", parse_dates=["iso_time"])
d = d[(d.synced == 1) & d.iso_time.between(START, END)]
# 0 MiB means the capture service was not running (restarted for config changes): a gap, not a value.
d["capture_rss_mib"] = d.capture_rss_mib.where(d.capture_rss_mib > 0)

ev = pd.read_csv(HERE / "events.txt", sep=" ", names=["date", "time", "what"])
ev["t"] = pd.to_datetime(ev.date + " " + ev.time.str.split(",").str[0])
ev = ev[ev.t.between(START, END)]
published = ev[ev.what == "outcome=publish"].t
recorded = ev[ev.what.isin(["outcome=retain", "outcome=discard"])].t

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(3, 1, figsize=(6.3, 3.4), sharex=True)
panels = [("temp_c", "SoC temp.\n(°C)"),
          ("cpu_pct", "CPU use\n(%)"),
          ("capture_rss_mib", "Memory\n(MiB)")]
for ax, (col, label) in zip(axes, panels):
    for t in recorded:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    for t in published:
        ax.axvline(t, color=ORANGE, lw=1, alpha=0.7, zorder=1)
    ax.plot(d.iso_time, d[col], color=BLUE, lw=1.6, zorder=2)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[-1].set_xlabel("Time on 10 October 2026")
axes[0].plot([], [], color=ORANGE, lw=1, label="Published visit")
axes[0].plot([], [], color=GRID, lw=2, label="Clip kept for review or discarded")
fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=7.5, labelcolor=INK,
           bbox_to_anchor=(0.5, 1.0))
fig.align_ylabels()
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(sys.argv[1])
print(f"{len(d)} rows, {len(published)} published, {len(recorded)} recorded; "
      f"temp {d.temp_c.min()}-{d.temp_c.max()} C, cpu max {d.cpu_pct.max()}%, "
      f"rss {d.capture_rss_mib.min()}-{d.capture_rss_mib.max()} MiB")
