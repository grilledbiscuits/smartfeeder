# Pi battery-only idle discharge, 2026-10-13

The Pi ran from the project's power manager after the supply switch on 13 October. The 2-second, fsynced runtime logger restarted automatically when that switch rebooted the Pi. This result is the **Pi idle baseline**, not full feeder idle: `birdcam-web`, health telemetry, and the runtime logger were active, while `birdcam-capture` failed to start because the intentionally unpowered VL53L1X did not answer on I²C. The camera was attached but did not record; solar input was not used for this desk test.

| Result | Value |
|---|---:|
| Last durable runtime checkpoint after logger restart | **61,311.442 s = 17 h 1 m 51.442 s** |
| Last checkpoint time after NTP sync | 2026-10-13 22:51:48.486 SAST |
| Runtime checkpoints | 30,507 |
| Largest gap between checkpoints | 2.979 s |
| Undervoltage/throttle flags in checkpoints | 0x0 throughout |
| Synced health samples during discharge | 2,027, all 0x0 throttle flags |
| Health temperature range | 44.3–53.5 °C |
| Capture process present in health samples | 0 of 2,027 |

`discharge.jsonl.gz` is the complete runtime log from boot ID `457e478b-a755-482a-8ecf-1f38270bea2f`, gzip-compressed without changing its JSONL content. The SHA-256 of the uncompressed log is `3be2379ae59981a125e7deaf239e44323e481573a742d63c60b0c7b591d96645`. `health.csv` contains only synchronized health samples during this discharge. `before-switch.jsonl` records the preceding mains-powered logger run, which stopped when the supply switch rebooted the Pi.

The logger's first wall-clock timestamps were wrong after reboot; NTP later advanced the clock by about 205 seconds. **Use `elapsed_s`, not the first timestamp or the filename, to calculate runtime.** The logger started about 10 seconds after battery boot, so 17 h 1 m 51 s is a conservative lower bound. The last row is `sample`, with no graceful `stopped` row. The exact power-cut instant was not observed externally and might be later than the last checkpoint. No voltage warning appeared in the sampled interval; a warning between the final sample and cutoff cannot be ruled out.

The Pi later came back online under boot ID `e4db1144-c925-4fdf-88c0-17e09e891f13`, with the runtime and telemetry services active. Its clock was initially unsynchronized on that boot too. The battery's starting state of charge was not independently logged, so this is a full-charge autonomy figure only if the power manager was fully charged at the switch. No watts were measured in this test.
