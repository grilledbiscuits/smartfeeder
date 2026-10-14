# Prepared battery inference discharge

**State:** installed on the Pi, disabled and stopped. Start only after the user switches the fully charged Waveshare Solar Power Manager (B) to battery-only output. Keep the solar charging input disconnected for comparison with the [Pi idle baseline](../battery-20261013/README.md).

The `birdcam-battery-inference.service` reuses the isolated 2026-10-12 replay checkout and its deployed 37-class INT8 model. It replays one of 20 labelled saved clips every **120 seconds**, looping until the battery cuts out. It exercises video decoding, preprocessing, classification, and outcome handling; it does **not** power or read the ToF sensor or record with the camera. The service conflicts with live capture so that service cannot add an uncontrolled workload. The web and health telemetry services remain in their baseline configuration.

The updated replay runner fsyncs each event row and its status file. It records boot ID, Pi uptime, inference duration, memory, temperature, clock, and throttle flags, and continues sampling after a voltage or thermal warning. The independent battery-runtime logger checkpoints every 2 seconds. The Pi output directory is `/home/grilledbiscuits/soak-20261012/ml/reports/battery-inference-next/`.

After the switch, start the inference service, verify its first event and the battery-runtime logger, then leave both running until cutoff. The measured result will be a **runtime at this fixed replay cadence**. Comparing it with the idle run can estimate average system power under the same nominal battery energy assumption; it cannot isolate the classifier chip's instantaneous watts or include camera recording power.
