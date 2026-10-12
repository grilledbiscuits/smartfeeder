# Pi replay soak, 2026-10-12

This is a **desk replay stability test**, not a field visit or independent
accuracy test. The Pi replays 20 labelled clips through the deployed 37-class
INT8 classifier for 72 cycles at 30-second intervals (1,440 events planned).
The ToF sensor is unpowered; the camera is not used by the replay runner.

- `pilot/`: complete one-cycle check, 20 events, 0 errors. All three files were
  checked before the full run began.
- `checkpoint/`: a snapshot of the **still-running** full soak at 14:31:44 SAST,
  460 events, 0 errors. Do not quote it as a completed 12-hour result.
- `events.jsonl`: one row per event, including processing time, memory, temperature,
  clock, throttle flags, classification, and action outcome.
- `status.json`: run state, counts, and SHA-256 hashes of the model, config,
  calibration, taxonomy, replay manifest, and runner.
- `runner.log`: startup warnings and uncaught errors from the process.

The completed full-run logs and separate software-trigger-to-first-camera-frame
measurement will be added after the soak finishes. No USB-C power measurement
is available from this laptop.
