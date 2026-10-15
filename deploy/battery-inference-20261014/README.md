# Pi battery discharge with saved-clip inference, 2026-10-14–15

The user confirmed that the Waveshare Solar Power Manager (B) was fully charged and had **no charging input** when this run began. The Pi replayed one saved clip every 120 seconds through video decoding, preprocessing, the deployed 37-class INT8 classifier and outcome handling. No live camera recording or ToF reading occurred. The web service, health telemetry and battery logger remained in their existing configuration; `birdcam-capture` was not running.

| Result | Value |
|---|---:|
| Last durable battery-runtime checkpoint | **66,688.959 s = 18 h 31 m 28.959 s** |
| Last checkpoint wall time, after clock sync | 2026-10-15 09:14:48.197 SAST |
| Runtime checkpoints | 33,182 |
| Largest checkpoint gap | 2.988 s |
| Inference events | 555 over 20 repeatedly replayed clips |
| Inference errors | 0 |
| Event spacing, median (range) | 120.00 s (117.62–121.53 s) |
| Event processing time, median (maximum) | 1.206 s (3.265 s) |
| Identification / action agreement with saved labels | 528/555 (95.1%) / 555/555 (100%) |
| Event temperature range | 37.0–56.0 °C |
| Event RSS range | 126,956–161,184 KiB |
| Undervoltage/throttle flags | `0x0` in all runtime, event and health samples |
| Synchronized health samples during this boot | 2,205; capture process present in 0 |

The first replay event was at Pi uptime 100.72 s (2026-10-14 14:44:47 SAST), and the last completed event was at uptime 66,578.92 s (2026-10-15 09:12:47 SAST). The final battery checkpoint was about 121 seconds after that event's trigger; the next scheduled event has no durable result. The Pi then lost power and later restarted under a different boot ID, `85079773-dc61-4745-9463-baa1f0ae2630`. Neither the runtime log nor the runner has a graceful stop record; `status.json` still says `running`. The checkpoint is therefore a **lower bound**, not a timestamp for the exact electrical cutoff. The logger started about 9 seconds after boot.

This lower bound exceeds the [full-charge idle baseline](../battery-20261013/README.md) of 17 h 1 m 51.442 s by **1 h 29 m 37.517 s (8.8%)**. Since the added inference work cannot itself establish a negative power cost, the two runtime trials do not isolate incremental inference power. Battery usable energy, conversion efficiency, device state and run-to-run variation were not directly measured. Under the same *nominal* 31.82 Wh output-energy assumption used for the baseline, this trial implies 31.82 Wh / 18.524711 h ≈ **1.72 W average at 5 V output**; this is a specification-based estimate, not a measured wattage. The repeated labelled clips test operational stability, not independent classifier accuracy.

`discharge.jsonl.gz` is the complete fsynced runtime log from boot ID `a59c48e5-248d-4d7a-a632-1a8ff6d3b0c9`. The SHA-256 of the uncompressed JSONL is `fc41e15f1c6adb21ef34fa25b3d5172f2d9c69d37bae4d9d960fb8dcc1de1962`. `events.jsonl`, `status.json` and `runner.log` are copied unchanged from the Pi. `health.csv` contains only the 2,205 rows from this discharge boot, selected between consecutive uptime resets in the Pi's appended health telemetry. The Pi clock began with an incorrect wall time (`09:57` in the runtime filename) before NTP advanced it; **calculate duration from `elapsed_s` and the boot ID**, not the early timestamps.
