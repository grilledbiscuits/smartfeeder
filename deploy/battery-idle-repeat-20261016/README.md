# Repeat Pi idle battery discharge, 2026-10-16–17

The user switched the Waveshare Solar Power Manager (B) to battery-only power after charging it. The switch rebooted the Pi into boot ID `0b6a9724-6420-49a5-9e73-64758071d4b2`. The 2-second, fsynced runtime logger, web dashboard, and health telemetry were active. Camera capture and both timed inference services were inactive; capture had been disabled before the test because its ToF sensor was intentionally unpowered. The camera did not record.

| Result | Value |
|---|---:|
| **Last durable checkpoint after logger start** | **62,025.447 s = 17 h 13 m 45.447 s** |
| Final checkpoint's wall-clock reading | 2026-10-17 07:24:02.887 SAST (clock unsynchronized at cutoff) |
| Runtime records | 30,867 (one start, 30,866 samples) |
| Largest checkpoint gap | 2.874 s |
| Undervoltage/throttle flags | `0x0` in every runtime and health sample |
| Health samples from discharge boot | 2,050; capture process present in 0 |
| Health temperature range | 31.1–59.4 °C |
| Difference from first idle lower bound | **+714.005 s = 11 m 54.005 s (+1.16%)** |
| Nominal average at 5 V output | 31.82 Wh ÷ 17.229291 h ≈ **1.85 W** |

This is a **lower bound**: the logger started after boot, and its last record is a `sample`, not a graceful stop. Electrical cutoff occurred later than that record. The user restored power after the test; the Pi booted under `d709894f-f5a9-42b2-a4f3-564bbfabf74a` and then restarted again under `a3f628b0-14d7-4dd3-b050-6c4770b79d15`. The first recovery boot's early wall-clock timestamp is wrong. `recovery.jsonl` documents that separate boot and is not part of the discharge duration.

The [first idle run](../battery-20261013/README.md) logged **17 h 1 m 51.442 s**. These two steady-state idle results differ by only 1.16%, so the repeat does **not provide evidence** that the first run was substantially undercharged. It cannot prove equal initial state of charge: the PMU's charge level and output energy were not measured. The first run had a failed capture service at startup; this run had capture disabled, but both had zero capture process samples during discharge. Clock synchronization was lost at Pi uptime 41,017 s (about 11 h 23 m into this run), so duration must be calculated from monotonic `elapsed_s`, not wall-clock subtraction.

The 1.85 W figure uses the same nominal 31.82 Wh output-energy assumption as the first report. It is a specification-based estimate, **not measured power**, and uses a lower-bound runtime. Actual usable capacity, conversion efficiency, and cutoff time remain unknown.

`discharge.jsonl.gz` is the complete fsynced runtime log, gzip-compressed without changing its JSONL content. The uncompressed SHA-256 is `9b88a73c6988920cb13b4464d17ab70751ebaf34e528a68c36bbe3d386b13aae`. `health.csv` contains only the 2,050 rows between this boot's uptime reset and the first recovery boot's reset. `recovery.jsonl` was copied unchanged from the Pi; its SHA-256 is `92720958d19bf66abbb7145138a72a1d040c3c393ead361769d0395ba5a69000`.
