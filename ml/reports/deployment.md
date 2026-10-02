# Deployment notes -- Raspberry Pi 4B

Target: Raspberry Pi 4B, with on-device CPU inference and hardware H.264 video
encoding. The deployed classifier uses ONNX Runtime. See `deploy/README.md` for
installation and service setup.

## Inference

* Benchmark the model on the Pi 4B. Laptop latency and MAC count do not predict
  ARM CPU latency reliably.
* Evaluate INT8 quantisation against FP32 accuracy and measured Pi 4B latency.
  Prefer ONNX Runtime with the XNNPACK execution provider when available:

      onnxruntime.InferenceSession(
          "birdcam_student.onnx",
          providers=["XnnpackExecutionProvider", "CPUExecutionProvider"],
      )

* Classify SAMPLED frames only, never every frame. A visit lasting seconds
  yields plenty; the track vote does the rest.
* Use the Pi 4B hardware H.264 encoder for the circular pre-roll buffer.
* Revisit the backbone if CPU latency is too high; `efficientnet_lite0` is a
  candidate for mobile CPU INT8, subject to accuracy measurement.

## Quantisation

`birdcam.export.quantize` performs INT8 post-training quantisation and reports
per-class accuracy delta. Read it before shipping either way: fine-grained
classes sit close together in feature space and INT8 can smear them, rarely
evenly across classes.

The calibration set MUST be real feeder crops once they exist. Web photographs
have different noise, blur and exposure statistics than a camera at a feeder.
