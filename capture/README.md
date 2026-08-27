# `capture/` — motion-triggered capture and classification

The on-device service for a Raspberry Pi 4B at the feeder. A PIR sensor fires,
the camera records a short clip, the existing classifier decides what is in it,
and the clip is either published to the dashboard or deleted.

It is the third sibling in this repo, alongside `ml/` (the classifier) and
`web/` (the dashboard), and the only one that touches hardware.

```
PIR (GPIO)  →  TriggerGate  →  Picamera2Recorder  →  BirdcamClipClassifier
                (cooldown,       (H.264 → mp4)         (ffmpeg frames →
                 queue cap)                             ONNX → decide() → vote())
                                                                │
                                          ┌─────────────────────┼─────────────────────┐
                                     should_record          uncertain            everything else
                                          │                     │                     │
                                    pending/ → var/media/   review/            deleted immediately
                                    + web.db.add_visit()    (capped)
```

---

## What it integrates with

Both of these are **fixed interfaces**. This package calls them; it does not
reimplement or modify them.

### The classifier — `birdcam.inference.Classifier`

```python
Classifier(cfg, novelty_scorer, temperature, range_prior)
    .decide(taxon_logits, sex_logits, features) -> Decision
    .vote(decisions) -> Decision
```

It takes **model outputs, not files**, so this package owns everything between
"an mp4 exists" and "logits exist": frame sampling, preprocessing, and the ONNX
session. Preprocessing reproduces `birdcam.train.augment.build_eval_transform`
exactly — resize the short side to `int(224 × 1.14)`, centre-crop to 224,
ImageNet mean/std — because `export/to_onnx.py` states the exported graph is
"preprocessed tensor in, two logit tensors out" and that normalisation stays in
the capture application.

**Classes of interest are not configured here.** The capture allowlist is
derived by `Classifier` itself from `ml/config/species.yaml`: Tier A species
plus the genus fallbacks whose genus contains a Tier A target. The service reads
`Decision.should_record` and does not second-guess it. Run
`python -m capture --check` to print the resulting allowlist.

### The dashboard — `web.db.add_visit`

`web/db.py` states the contract in its own docstring: the feeder process saves
its media and calls `add_visit()`. There is **no upload endpoint** in `web/` —
no REST route, no auth, no multipart — and this package does not add one. It
writes the clip into `var/media/videos/`, the thumbnail into
`var/media/images/`, and inserts one row. SQLite is already in WAL mode so the
dashboard can read while this service writes.

---

## Wiring

### HC-SR501 PIR

| HC-SR501 | Pi 4B physical pin | BCM |
|---|---|---|
| `VCC` | 2 or 4 (5 V) | — |
| `OUT` | 7 | **GPIO4** |
| `GND` | 6 | — |

> **Check `OUT` with a multimeter before connecting it.** The HC-SR501 is
> powered at 5 V, and the common modules level-shift their output to 3.3 V —
> which is what makes them safe on a Pi. Some clones do not, and 5 V on a GPIO
> input will damage the SoC. Measure the pin while triggered; if it reads 5 V,
> put a divider (e.g. 10 kΩ / 20 kΩ) between `OUT` and the Pi.

On-board adjustments:

* **H/L jumper → H (repeatable trigger).** In `L` the module will not re-assert
  until its timer fully lapses, which loses a bird that lands during the gap.
* **`Tx` (time delay) → fully anticlockwise (minimum, ~3 s).** The software
  cooldown owns retrigger suppression; a long hardware delay only hides motion
  from the service.
* **`Sx` (sensitivity) → start around the middle** and tune outdoors. A feeder
  in direct sun will false-trigger on thermal gradients.

The GPIO pin is configurable (`gpio.pin`, BCM numbering).

### Camera Module 3

Ribbon into the **CAMERA** port (nearest the audio jack on a Pi 4B), contacts
facing away from the Ethernet port. Verify before installing anything else:

```bash
libcamera-hello --list-cameras
```

Camera Module 3 is an IMX708 behind libcamera. The legacy `picamera` library
does not support it at all; `picamera2` is required.

---

## Install

### 1. System packages

`picamera2` and `gpiozero` are **apt packages, not pip packages** — they are
built against the system libcamera. The virtualenv must be able to see them.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-gpiozero python3-lgpio ffmpeg git
```

### 2. Checkout and virtualenv

Deploy from a clone rather than a wheel: the service needs `ml/config/`,
`ml/data/export/` and the `web/` package at runtime.

```bash
sudo git clone <repo-url> /opt/smartfeeder
cd /opt/smartfeeder
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r capture/requirements-pi.txt
```

`--system-site-packages` is load-bearing. Without it the venv cannot see
`picamera2` or `gpiozero` and the service will not start.

### 3. Model artefacts

`ml/data/` is gitignored, so the ONNX export and its sidecar must be copied to
the Pi by hand:

```
ml/data/export/birdcam_student.onnx
ml/data/export/birdcam_student.json      # class order + image size; ships together
ml/reports/operating_points_finetuned.json   # temperature + fitted thresholds
```

> **See "Known issues" below before doing this.** The export currently in the
> repo does not match the calibration, and the service will refuse to start on
> that pairing.

### 4. Config

```bash
cp capture/config/capture.example.yaml capture/config/capture.yaml
$EDITOR capture/config/capture.yaml
.venv/bin/python -m capture --check
```

`--check` builds every component without arming the sensor and prints the
resolved settings, free space, and the capture allowlist. It exits non-zero if
the model artefacts are unusable.

### 5. systemd

```bash
sudo cp deploy/birdcam-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now birdcam-capture
```

Adjust `User`, `WorkingDirectory` and `ExecStart` if the checkout is not at
`/opt/smartfeeder`. The unit sets `SupplementaryGroups=video gpio` — both are
needed, and it deliberately does **not** set `PrivateDevices=yes`, which would
hide `/dev/video*` and `/dev/gpiochip*` and produce a confusing failure.

```bash
systemctl status birdcam-capture
journalctl -u birdcam-capture -f
sudo systemctl restart birdcam-capture
```

The unit restarts on any exit, with a 5-failures-in-10-minutes limit so a
crash loop cannot hammer the SD card. Clear it with
`sudo systemctl reset-failed birdcam-capture`.

---

## Testing without hardware

Everything except the sensor and the camera runs unchanged off-Pi.

**One clip through the whole decision pipeline:**

```bash
python -m capture --classify path/to/clip.mp4
```

Records (by copying), samples frames, classifies, applies the keep/discard
rule, and publishes or deletes — then prints the full record.

**The service, with a fake PIR and a replayed clip:**

```bash
python -m capture --mock --replay path/to/clip.mp4 --triggers 3 --interval 1
```

This runs the real state machine, cooldown, spool, and publisher. With a 30 s
cooldown, triggers 2 and 3 are dropped and logged — which is the behaviour to
check.

**The logic tests** (no model, no hardware, no dashboard):

```bash
pytest capture/tests -q
```

They cover the state machine, the keep/delete/publish rule, the spool's durable
queue, backoff and retry, the publisher contract, and every fatal config
cross-check.

---

## Config reference

`capture/config/capture.example.yaml` is the complete reference and every key is
commented there. There are **no defaults in Python** — a missing key is a
startup error naming the key. Highlights:

| Key | Default | Notes |
|---|---|---|
| `gpio.pin` | `4` | BCM numbering |
| `gpio.warmup_seconds` | `60` | HC-SR501 settling; without it a restart fires a burst of empty recordings |
| `camera.width` / `height` | `1280×720` | Must be even (H.264 4:2:0). The classifier crops to 224 px, so more pixels are discarded, not used |
| `camera.bitrate_kbps` | `4000` | ~4 MB per 8 s clip. The Pi 4B keeps the hardware H.264 encoder the Pi 5 dropped |
| `capture.clip_seconds` | `8.0` | |
| `capture.cooldown_seconds` | `30.0` | Starts at **admission**. Must be ≥ `clip_seconds`; enforced at load time |
| `capture.on_busy` | `queue` | `queue` or `drop`, for a trigger arriving mid-event |
| `capture.max_queued` | `1` | Events waiting behind the one in flight |
| `storage.min_free_mb` | `1024` | Recording stops rather than filling the card |
| `storage.max_pending_clips` | `200` | Hard ceiling on **unconfirmed** clips |
| `storage.max_review_clips` | `100` | Abstained clips; these *are* evictable, oldest first |
| `storage.delete_after_publish` | `true` | |
| `classifier.enabled` | `true` | `false` records and retains everything unclassified |
| `classifier.sample_fps` / `max_frames` | `2.0` / `12` | Sampled frames only, never every frame |
| `classifier.providers` | XNNPACK, CPU | XNNPACK has the NEON INT8 kernels for the Pi 4B |
| `classifier.novelty.enabled` | `false` | See "Known issues" |
| `publish.retain_uncertain` | `true` | Keep abstained clips for review; never published |
| `publish.escalate_after_attempts` | `5` | **Not** a give-up count — see below |
| `logging.format` | `text` | `json` for one object per line |

### Secrets

Any config value may contain `${VAR}` or `${VAR:-fallback}`, expanded from the
environment at load time. An unset `${VAR}` with no fallback is a **fatal
error**, not an empty string. Set them in the systemd environment file:

```bash
sudo install -m 600 /dev/null /etc/default/birdcam-capture
echo 'SOME_TOKEN=…' | sudo tee -a /etc/default/birdcam-capture
```

The same-host publisher needs no credentials, so nothing requires one today.
`capture/config/capture.yaml` is gitignored for this reason.

---

## Behaviour worth knowing

**A clip whose publication is unconfirmed is never deleted.** Not on retry
exhaustion, not to free space. `publish.escalate_after_attempts` only controls
when a failure escalates from `WARNING` to `ERROR`; retries continue
indefinitely at `backoff_max_seconds`. What bounds disk use is
`storage.max_pending_clips`, which stops *new recordings* rather than
discarding evidence — so a dashboard that has been down for a week leaves a
full spool and a log full of errors, not a gap.

**Crash recovery.** `work/` is cleared at startup (an unfinalised H.264 stream
is not a clip). `pending/` is re-queued and drained before the sensor is even
armed. A pending clip whose sidecar is missing or corrupt is re-adopted, not
deleted.

**Concurrency.** The cooldown is stamped when a trigger is *admitted*, not when
recording starts. When nothing is busy those are the same instant; when the
previous event is still classifying they are not, and stamping at admission is
what stops a burst of HC-SR501 re-triggers queueing up and recording the same
visit several times.

**Failure isolation.** Camera-busy, GPIO-busy, disk-full, ffmpeg failure, a bad
frame, a locked database — each is logged against the event id and the service
returns to idle. Nothing propagates out of the worker loop.

---

## Known issues

**The model artefacts in this repo cannot currently be deployed together.**
`ASSUMPTIONS.md` A26: `ml/data/export/birdcam_student.onnx` is a
*frozen-feature* export, while `config/taxonomy.yaml`'s thresholds and
`reports/operating_points_finetuned.json`'s temperature were fitted on the
*fine-tuned* checkpoint. Both artefacts are individually valid, which is what
makes the pairing dangerous: nothing fails, the labels are simply wrong. The
service detects this and **refuses to start**. Re-export from
`student_best.pt`, or set `classifier.allow_artefact_mismatch: true` to
override deliberately.

**The open-set failsafe is off.** Without it the softmax always returns one of
the 62 taxon classes, so a squirrel is reported as the least-bad bird rather
than `unknown`. It is off because the deployment bundle cannot reconstruct the
kNN scorer: the ONNX graph emits logits only, and kNN needs the 1,280-d pooled
features from the *frozen* backbone plus its reference vectors (A25, A26).
The seam is in place — `classifier.novelty` — and `scorer: energy` works
**today** with no extra artefacts, scoring the logits the graph already emits
(AUROC 0.927, 68.1 % of OOD caught at 5 % false-alarm rate, threshold −8.4859,
from `reports/open_set.json`). Weaker than kNN, but a real gate rather than
none.

**The guild rollup emits a label that is not in the label space.**
`Classifier.decide` builds guild fallbacks as `f"{guild}_indet"`, and
`taxonomy.yaml` maps ten families to the guild `non_target` — but only
`nectarivore_indet` exists as a class. Any non-target bird that rolls up to
guild therefore returns `non_target_indet`, which is in no class list.
Observed on a real clip during development. It is *behaviourally* safe here —
`is_capture_target` is `False`, so the clip is discarded correctly — and this
package renders it as "Non-target bird" rather than nonsense. But it lives in
`inference.py`, which this package treats as a fixed interface, so it is
reported rather than patched. The genus branch already guards this case with
`if slug in self.cfg.taxon_class_index`; the family and guild branches do not.

**One duplicate-row window.** A crash between a successful `add_visit()` and
the sidecar recording its `visit_id` re-publishes that clip on restart,
producing one duplicate dashboard row. Closing it properly needs a unique
constraint on `visits`, which is a schema change to the other side of a fixed
interface.

**Expect under-triggering, not squirrels.** `ASSUMPTIONS.md` A27, on 11,050
real feeder frames: roughly 42 % of the timeline has a bird present, but only
15.7 % would commit video. The keep/discard logic here is correct; the model
behind it currently misses most genuine visits. `publish.retain_uncertain` is
on by default so the abstained clips — the ones worth a human look — survive
locally instead of being deleted.
