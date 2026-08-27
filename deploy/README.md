# Deploying the full pipeline on a Raspberry Pi 4B

End-to-end runbook for a fresh board: PIR-triggered capture, on-device
classification, and the dashboard, both under systemd.

This is the *procedure*. The reasoning behind each component lives elsewhere and
is not repeated here:

* `capture/README.md` — wiring, the config reference, and what the service does
* `ml/reports/deployment.md` — why the Pi 4B trades a worse accelerator for a
  hardware H.264 encoder
* `ml/ASSUMPTIONS.md` — A26 (artefact pairing) and A27 (under-triggering)

Target: Raspberry Pi 4B, 64-bit Raspberry Pi OS **Lite** (Bookworm), Camera
Module 3, HC-SR501 PIR. No desktop, CLI only.

---

## What does not arrive with a clone

Two categories, and both bite silently:

* **`ml/data/` and `ml/reports/` are gitignored.** The ONNX export, its
  metadata sidecar and the fitted operating points must be copied by hand.
* **`capture/config/capture.yaml` is gitignored.** It is per-deployment and may
  carry `${VAR}`-expanded secrets. Copy `capture.example.yaml` on the device.

Everything else — `ml/config/`, `ml/src/`, `capture/`, `web/`, this unit file —
is tracked.

---

## 1. System packages

`picamera2` and `gpiozero` come from apt, not pip: they are built against the
system libcamera, and the legacy `picamera` library cannot drive Camera Module 3
at all.

```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-gpiozero python3-lgpio python3-venv ffmpeg git
```

Confirm the architecture. `onnxruntime` publishes no 32-bit ARM wheel, so a
`armv7l` result means reflashing with the 64-bit image before going further:

```bash
uname -m   # must print aarch64
```

## 2. Verify the hardware first

```bash
rpicam-hello --list-cameras
```

(`libcamera-hello` on older Bookworm images.) An IMX708 should be listed.

**Measure the PIR's `OUT` pin with a multimeter while triggered before
connecting it.** The HC-SR501 is powered at 5 V; most modules level-shift their
output to 3.3 V, some clones do not, and 5 V on a GPIO input damages the SoC.
Wiring and the on-board jumper settings are in `capture/README.md`.

## 3. Clone and build the virtualenv

```bash
sudo git clone https://github.com/grilledbiscuits/smartfeeder.git /opt/smartfeeder
```

```bash
sudo python3 -m venv --system-site-packages /opt/smartfeeder/.venv
```

`--system-site-packages` is load-bearing. Without it the venv cannot see
`picamera2` or `gpiozero`, and the service will not start.

```bash
sudo /opt/smartfeeder/.venv/bin/pip install -r /opt/smartfeeder/capture/requirements-pi.txt
```

That requirements file includes Flask, so this one virtualenv serves both the
capture service and the dashboard.

## 4. Copy the model artefacts

Three files, ~24 MB. From a workstation checkout that has them:

```bash
scp ml/data/export/birdcam_student.onnx ml/data/export/birdcam_student.json ml/reports/operating_points_finetuned.json <pi>:/tmp/
```

On the Pi:

```bash
sudo mkdir -p /opt/smartfeeder/ml/data/export /opt/smartfeeder/ml/reports && sudo mv /tmp/birdcam_student.{onnx,json} /opt/smartfeeder/ml/data/export/ && sudo mv /tmp/operating_points_finetuned.json /opt/smartfeeder/ml/reports/
```

The graph and the calibration must describe **one model**. `birdcam_student.json`
should carry `"checkpoint_sha": "a71e95cca471…"`, matching the SHA in the
operating points' `source`. Step 6 verifies this; A26 explains why a mismatch is
dangerous rather than merely wrong. If the export predates provenance stamping,
regenerate it on a machine with torch:

```bash
python -m birdcam.export.to_onnx --checkpoint ml/data/checkpoints/student_best.pt
```

## 5. Service user, runtime directory, config

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin birdcam
```

```bash
sudo mkdir -p /opt/smartfeeder/var && sudo chown -R birdcam:birdcam /opt/smartfeeder/var
```

`var/` is the only path either service writes; both units pin `ReadWritePaths`
to it.

```bash
sudo cp /opt/smartfeeder/capture/config/capture.example.yaml /opt/smartfeeder/capture/config/capture.yaml
```

**Decide the open-set gate before arming anything.** As shipped it is the
`energy` scorer at `-5.669`. Measured on field frames, that discards 84% of
adult Amethyst Sunbird frames and votes the whole visit `unknown` — the species
sits above the empty-feeder distribution in energy, so no threshold separates
them. For a first deployment, turning it off favours catching birds over
suppressing empty frames, which is the direction A27 argues for:

```bash
sudo sed -i '/^  novelty:/,/^    reference:/ s/enabled: true/enabled: false/' /opt/smartfeeder/capture/config/capture.yaml
```

With the gate off, empty frames abstain to `uncertain` and are shelved in
`review/` under `storage.max_review_clips` rather than deleted, so the first
day's footage tells you what the feeder actually produces. Leave the gate on if
filling the card matters more than losing that species.

## 6. Pre-flight check

```bash
cd /opt/smartfeeder && sudo -u birdcam .venv/bin/python -m capture --check
```

Builds every component *without* arming the sensor: resolved settings, free
space, and the capture allowlist. **It must exit 0.** A non-zero exit names the
problem — a missing sidecar, a class-order mismatch, or a graph and calibration
from different checkpoints.

One warning is expected and harmless: the stock `onnxruntime` wheel has no
`XnnpackExecutionProvider`, so it falls back to `CPUExecutionProvider`, which
still uses NEON via MLAS.

## 7. Install both services

```bash
sudo install -m 0644 /opt/smartfeeder/deploy/birdcam-capture.service /etc/systemd/system/
sudo install -m 0644 /opt/smartfeeder/deploy/birdcam-web.service /etc/systemd/system/
```

Both unit files are tracked in `deploy/`. The dashboard binds to all LAN
interfaces by default because the target is headless. It has no authentication,
so do not expose port 5000 through router port forwarding. Optional environment
overrides can be placed in `/etc/default/birdcam-web`; do not enable
`BIRDCAM_WEB_DEBUG` while it is listening on a non-loopback address.

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now birdcam-capture birdcam-web
```

This is a Flask development server on a LAN with no authentication. Do not port
forward it.

## 8. Verify end to end

```bash
journalctl -u birdcam-capture -f
```

Startup drains any spooled backlog, then waits `gpio.warmup_seconds` (60 s by
default) for the HC-SR501 to settle before logging `armed; waiting for motion`.
Skipping that warmup produces a burst of empty recordings on every restart.

Wave a hand in front of the sensor. A recorded visit logs, in order:
`motion admitted` → `recorded 8.0s` → `capture decided` → `publish confirmed`
with a `visit_id`. The row appears at `http://<pi-ip>:5000` immediately — both
processes share `var/feeder.db` in WAL mode.

`outcome=discard` with `allowlisted=False` is **correct behaviour**, not a
fault: Tier B and C birds exist so the model can recognise a bystander without
committing storage for it.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `could not claim GPIO4` | a second copy of the service is running, or the `gpio` group is missing |
| `picamera2 is not importable` | venv built without `--system-site-packages`; rebuild it |
| `could not open the camera` | libcamera allows one client — check nothing else holds it |
| `--check` exits 1 on artefacts | wrong or stale ONNX copied; the sidecar SHA must match the operating points |
| `ffmpeg is not on PATH` | `sudo apt install -y ffmpeg` |
| every event ends `discard` | expected with the novelty gate on; check `is_unknown` in the log |
| unit gives up after 5 restarts | `sudo systemctl reset-failed birdcam-capture` |

Logs: `journalctl -u birdcam-capture -f`, `journalctl -u birdcam-web -f`. Set
`logging.format: json` in `capture.yaml` for one object per line.

---

## Known limitations of this deployment

**The graph is FP32.** `ml/reports/deployment.md` calls INT8 mandatory rather
than optional on a Pi 4B CPU, but `birdcam_student_int8.onnx` was quantised from
the superseded frozen-feature export. It ships without a sidecar, so pointing
`classifier.onnx_path` at it fails at startup instead of silently running the
wrong graph. Re-run `birdcam.export.quantize` against the current export once
the board has produced real latency numbers.

**Expect under-triggering, not false alarms.** A27, on 11,050 real feeder
frames: roughly 42% of the timeline has a bird present, but only 15.7% commits
video. `publish.retain_uncertain` is on by default so abstained clips survive in
`review/` — they are the local evidence of what is being missed, and are worth
reviewing after the first day.
