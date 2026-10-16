# Pi battery discharge with camera recording and inference, 2026-10-15–16

The user switched the fully charged Waveshare Solar Power Manager (B) to battery-only power on 15 October. The Pi recorded a fresh 8-second, 1280 × 720 H.264 clip every 30 seconds, classified 12 frames with the deployed INT8 model, and deleted each completed clip. A mock trigger kept the unpowered ToF sensor out of the test. The camera faced the desk, so labels are workload evidence, not accuracy evidence. The normal web and health services and the independent 2-second, fsynced battery-runtime logger were active; live capture and the earlier saved-clip replay service were not running.

| Result | Value |
|---|---:|
| Test start | 2026-10-15 17:36:28.691 SAST, Pi uptime 117.56 s |
| Last durable runtime checkpoint | 2026-10-16 07:47:38.613 SAST, Pi uptime 51,177.843 s |
| **Battery workload runtime lower bound** | **51,060.283 s = 14 h 11 m 0.283 s** |
| Battery boot-to-last-checkpoint lower bound | 51,177.843 s = 14 h 12 m 57.843 s |
| Runtime log records | 25,450 (one start, 25,449 samples) |
| Largest runtime-checkpoint gap | 3.417 s |
| Completed recording and inference events | 1,702; zero logged errors; 20,424 frames scored |
| Event start spacing | 30 s for all 1,701 intervals |
| Processing time per event | median 10.424 s; maximum 13.154 s |
| Recording / inference time per event | medians 8.778 s / 1.662 s |
| Total completed clip bytes written | 3.643 GB (decimal), deleted after each event |
| Event temperature | 57.0–62.8 °C |
| Health samples from this boot | 1,693, all synchronized |
| Pi throttle flags | `0x50000` at the first runtime and health samples and throughout; current-condition bits zero in every sample |

The final completed event began at 07:46:58. The next event began at 07:47:28 and left an unfinished clip and zero-byte frame files when power failed. The last battery checkpoint was 07:47:38, so electrical cutoff occurred **after** that point and before the user's later report that the battery had died. `status.json` still says `running` because there was no graceful shutdown. The Pi restarted under boot ID `077131a9-df5f-4363-8f2e-0471ae0c8521` after USB power was restored. The unfinished desk clip and frame files were deleted after the raw logs had been copied; no recording from this run remains in `scratch/`.

`0x50000` means an undervoltage event and throttling had occurred at some point since that boot, already before the test began. The low current-condition bits were clear whenever sampled. These latched flags cannot locate the earlier event or prove that voltage stayed healthy between 2-second samples. FFmpeg emitted timestamp/DTS warnings in `runner.log`, but every completed cycle produced 12 scored frames and no event error. The recording loop consumed about 34.9% of the test interval (sum of event processing times divided by the runtime lower bound).

The workload lower bound is **2 h 50 m 51.159 s (16.7%) shorter** than the [idle baseline](../battery-20261013/README.md), and **4 h 20 m 28.676 s (23.4%) shorter** than the [saved-clip inference run](../battery-inference-20261014/README.md). Those trials differ in load and battery conditions; the saved-clip run lasted longer than idle, showing that runtime differences alone do not isolate component power. Under the same *nominal* 31.82 Wh output-energy assumption as the idle report, 31.82 Wh / 14.183412 h ≈ **2.24 W average at 5 V output**. This is a specification-based estimate using a lower-bound runtime, **not measured watts**. Actual usable energy, conversion efficiency, and exact cutoff time were not measured.

`events.jsonl`, `status.json`, and `runner.log` were copied unchanged from the Pi. `discharge.jsonl.gz` is the complete fsynced runtime log from boot ID `ccedd592-57c0-4084-a217-0a079f8e03f2`, gzip-compressed without changing the JSONL content. Its uncompressed SHA-256 is `afb98bd8988862a9dc9fdab440dc1acea0f9591ff45bbd53de4b9dd51873b86e`. `health.csv` contains only the 1,693 samples between this boot's uptime reset and the following USB-powered boot. The Pi clock was corrected shortly after battery boot, so calculate durations using uptime and `elapsed_s`; do not subtract the first and last wall-clock timestamps.
