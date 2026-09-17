# In flight — isolated Pi replay soak (updated 2026-09-16)

**Read first — completed September 17:** The isolated Pi soak finished normally
at **02:52:27 SAST**, after 6h 10m 30s: 741 events, zero runtime errors. All 39
clips produced the same label/outcome across 19 cycles. Capture actions remain
correct on 34/39 distinct development clips; the same five failures persist.
Live service is active with zero restarts since September 16 19:55:37.
**Do not deploy the candidate to the field yet.** Earlier RUNNING/PID instructions
below are historical; the soak has finished. See completion results at the end.

## Codex verification — 2026-09-16

Local verification completed; no remote commands, service changes, model changes,
or threshold writes were performed. The Pi state below is the previous handover's
observation, not a fresh remote check.

- Extraction finished after the pause: 41,056 ID and 2,486 OOD rows. Both files
  load successfully, match the current ordered image IDs and taxon class order,
  contain finite features/logits, and match the actual checkpoint SHA
  `0311dc3cf26ab84b48d858dcc94bf58d2d0929fcecb3264b20a4807201a0c47c`.
  The data loader reports 83 unmatched web images excluded; the counts above
  describe the loader's selected corpus, not all manifest rows.
- The INT8 sidecar matches that checkpoint and current class order. Its recorded
  ONNX SHA matches the actual graph. This verifies identity, not predictive quality.
- `python -m pytest -q`: **283 passed**. Ruff on ML, capture and deploy passed;
  `bash -n` passed for both deployment scripts. Scripts have not been executed.

### Resolve before writing production thresholds or deploying

1. `eval/thresholds.py::_calibrate_and_measure` fits temperature and thresholds
   using all raw logits. Production inference suppresses `empty_feeder` and may
   apply a site prior before softmax. Calibration must use the same transformation;
   suppressed-label validation examples also need an explicit evaluation policy.
   Its trigger curve uses kNN, whereas the intended service uses energy. Do not
   treat that curve as validation of the intended deployed decision path.
2. The rolling empty gate can absorb occupied scenes. Reproduced with the existing
   synthetic scene fixture: initialize 48 empty observations, then an occupied
   frame correctly returns `is_empty=False`; observe that occupied frame 48 times,
   and the same frame returns `is_empty=True`. The classifier observes incoming
   clip frames before testing them. PIR-triggered samples do not establish that
   the buffer is mostly empty. The passing tests and historical static-background
   percentages do not establish safety for this rolling deployment behavior.
   A trusted background strategy and sequence-level validation are required.

The ordered list below is the original plan, now conditional on these blockers.
Step 1 is verified complete; do not rerun extraction unnecessarily.

## State right now

**The Pi is safe and unchanged.** `birdcam-capture` is active on `sunfeed`
(192.168.0.224, user grilledbiscuits, SSH key auth; `.local` does not resolve)
running the **August 62-class FP32** build. Nothing under `/opt/smartfeeder`
has been modified. Today's code is staged separately at `~/birdcam-next` on the
Pi, plus 10 replay clips in `~/birdcam-next/replay/`.

sudo is now passwordless for ALL commands (main sudoers was edited). Narrow it
back to the birdcam-capture restart rule after deploying.

## Done today (committed)

- Retrained: 37 classes, best epoch 13, Tier A recall 0.7341, ECE 0.0324.
- `empty_feeder` suppressed at inference; emptiness decided geometrically
  (`capture/emptygate.py`, 98.9% / 97.5%).
- INT8 with FIELD calibration: field 0.8857 -> 0.8647, web 0.7300 -> 0.6955.
  Field calibration is best or tied on every metric vs mixed and web.
- Pi latency measured: INT8 52 ms/frame, 0.62 s per clip. XNNPACK absent, no cost.
- Camera hardware check passed: record() works after peek().
- `empty_gate.placement` switch, default `after_record`: peek costs ~2 s on the
  Pi 4B, which delayed every recording. Fixed an empty-clip-retained bug.
- `deploy/swap_in.sh` (backup, --check as birdcam, auto-rollback) and
  `deploy/rollback.sh`. WRITTEN BUT NEVER RUN.

## Remaining, in order

1. **Wait for / re-run extraction** (was ~75% at pause; writes
   `ml/data/embeddings/finetuned/student_best_{id,ood}.npz`, now includes field):
       python -m birdcam.eval.extract --checkpoint student_best.pt --batch-size 64
2. **Refit per-class thresholds** (writes operating_points_finetuned.json AND
   taxonomy.yaml per_class_thresholds, which is what inference reads):
       python -m birdcam.eval.thresholds --checkpoint student_best.pt --target-precision 0.8 --write-config
3. **Refit the novelty energy threshold.** The service scores energy on RAW
   logits at T=1, before prior and temperature. Old value -5.669 = 90th
   percentile of energy over FIELD bird frames (reject 10%). Recompute on val
   field bird frames. Then compare against INT8 logits
   (`scratchpad/int8_val.npz` if it survived, else recompute) — the Pi runs INT8.
4. **Build the deploy config**: `capture/config/capture.example.yaml` with
   `onnx_path: ml/data/export/birdcam_student_int8.onnx`, the new novelty
   threshold, and `empty_gate.placement: after_record`.
5. **Stage artefacts** to `~/birdcam-next/ml/data/export/`
   (`birdcam_student_int8.onnx` + `birdcam_student_int8.json`) and
   `~/birdcam-next/ml/reports/operating_points_finetuned.json`; re-sync code.
6. **Replay test** on the Pi (needs a staging config with the replay recorder):
       python deploy/replay_sequence.py --config <staging yaml> --clips replay/
7. **Deploy** — run on the Pi in `tmux`/`nohup`, never over a bare SSH
   session that can drop mid-swap:
       bash deploy/swap_in.sh ~/birdcam-next
8. **Soak**, then check memory, errors, `vcgencmd get_throttled`.

## Desk-Pi continuation — Codex, 2026-09-16 (in progress)

User explicitly authorized SSH testing on the desk Pi and requested a durable
handover for Claude. SSH key access works with `grilledbiscuits@192.168.0.224`.
Sandbox network access requires escalation; connection was approved. Do not
mistake a sandbox socket denial for the Pi being offline.

### Fresh remote observations

- Host `sunfeed`; `birdcam-capture` active, NRestarts=0, active since 19:55:37 SAST.
- Temperature 38.9 C, available RAM 7344 MB, swap unused.
- `vcgencmd get_throttled`: `0x80000` (raw observation; keep historical flags
  distinct from current flags when interpreting).
- Ten staged replay clips: four empty, five Southern Double-collared, one Cape
  Bulbul. No broad coverage of species or hands/OOD. Manifest has no source image
  IDs/split provenance: do not claim these are an untouched holdout.
- Live `/opt/smartfeeder` and live service remain untouched.

### Local changes under verification

- Production empty-gate builder now requires an explicit verified reference.
  Added `TrustedEmptyGate`, which never updates from PIR frames; legacy rolling
  `EmptyGate` retained for experiments. Without a reference, the production gate
  is bypassed. Example defaults off, placement after_record. Actual field camera
  reference and independent sequence validation still required.
- Fixed capture frame sampling's portrait scaling (short side now stays 256).
- Replay empty verdict uses DISCARD outcome, not whether recording happened;
  errors cannot pass as successful empty rejections. `record.empty` identifies
  the gate, including after-record gating.
- Added `Classifier.probability_logits()` shared by inference and calibration.
- New `ml/src/birdcam/eval/calibrate_capture.py` writes isolated candidates only.
  Fits on validation INT8 logits, capture preprocessing, site prior and suppressed
  classes; excludes suppressed true labels from temperature loss, but includes
  empty frames when measuring false target predictions. Energy uses raw logits
  at T=1, threshold is validation field-bird 90th percentile. Per-species fitting
  requires actual argmax and energy acceptance. Genus precision is not guaranteed.
- New regression tests cover persistent-bird absorption prevention, missing
  reference config, portrait video sampling, replay verdicts, transform parity.

### Running command / saved outputs

From repo root:

```bash
PYTHONPATH=ml/src:. .venv/bin/python -m birdcam.eval.calibrate_capture \
  --onnx ml/data/export/birdcam_student_int8.onnx \
  --out ml/reports/desk-20260916 \
  > ml/reports/desk-20260916/calibrate.log 2>&1
```

6247 validation images. Cache is keyed by graph hash, ordered image IDs and image
byte hashes, preprocessing code hash and label order. Cache has no pickle arrays.
This is accuracy/calibration work, not a laptop latency benchmark. Temporary old
`int8_val.npz` deliberately NOT reused: its script used timm's transform rather
than the capture transform and did not stamp graph provenance.

Initial code transfer (no live changes):

```bash
rsync -a --exclude='__pycache__' capture deploy ml/src ml/config \
  grilledbiscuits@192.168.0.224:birdcam-next/check-transfer/
```

This transfer flattens those source basenames into check-transfer/{capture,deploy,
src,config}; assemble into a separate replay root before use. No calibration or
model has been installed from this transfer yet.

### Results and active overnight run — updated ~20:42 SAST

**Candidate is NOT approved for field deployment. Live service is unchanged.**

Fresh INT8 capture calibration completed successfully:

- Temperature 1.390693993330419; energy threshold -4.763944288110461.
- Fit: 6247 validation frames, temperature on 6166 nonsuppressed labels; energy
  percentile on 1406 field bird frames. Empty labels excluded only from the
  temperature loss, not from false-prediction diagnostics.
- Thresholds, config and image/model fingerprints:
  `ml/reports/desk-20260916/{operating_points.json,taxonomy.yaml,replay.yaml}`.
  These are candidate files: local `ml/config/taxonomy.yaml` was NOT overwritten.
- 0/81 validation empty frames would publish, but the Pi replay contradicted any
  inference that this solves empty scenes. These are fitting diagnostics only.
- Exact new per-class thresholds are in that JSON; all six reached >=0.8 precision
  under the validation argmax+energy criterion. Genus rollup and unseen data can
  still fail. The older `eval.thresholds` CLI remains unsuitable for this policy.

Pi staging root: `/home/grilledbiscuits/birdcam-next/desk-test`.
It contains copies of the candidate graph, taxonomy, operating points, replay
config, capture/ML/deploy code and unchanged web code. It uses its OWN var/
spool, database and media. The runtime Python is `/opt/smartfeeder/.venv/bin/python`
(read-only reuse of dependencies). No packages installed and no service restart.

Replay tests:

1. Gate OFF: all six bird clips identified correctly; all FOUR empty clips
   wrongly published. Report: `desk-test/ml/reports/replay_results.json`.
2. Fixed reference: extracted first frame from `replay/00_empty1.mp4` into
   `ml/reports/reference.jpg`; inspected full frame locally: feeder/port visible,
   no bird visible. Gate rejected reference clip + TWO of the THREE other empty
   clips. All six bird clips identified correctly. The remaining empty4 clip
   still published via genus rollup. Reference clip MUST NOT count as independent
   validation. Report: `replay_fixed_results.json`.
3. Broader smoke: reconstructed ALL 36 nonempty field-test clip groups, first up
   to 24 labelled frames per group, at 2 fps. No original footage edited. Built
   with `deploy/build_field_replays.py --out ml/reports/desk-20260916/field-replay`.
   Manifest stamps image IDs and byte hashes. Frames are re-encoded low-resolution
   extracts, not original FHD clips. Added the three non-reference empty clips
   from the original replay manifest: **39 total** in Pi `soak-clips/`.
4. Full pipeline smoke completed: 39 events, 0 runtime errors, **34/39 correct
   publish/discard actions**. The raw identification rubric scored 33/39 because
   it also counts a hand classified unknown as non-exact, although discarding it
   is the desired action. RSS 99,668–161,016 KiB; sampled temperatures 49.7–60.9 C;
   throttle word remained 0x80000. These are real Pi observations, not laptop timing.

Action errors in broad smoke (all require follow-up, not automatic threshold tuning):

- Original empty4 -> genus fallback -> published.
- Amethyst clips 3 and 4 -> family fallback -> discarded.
- Southern Double-collared clips 21 and 3 (20260828_155630 session) -> other_animal
  -> discarded.
- Hand clip3 -> unknown -> correctly discarded (not an action error).

### Overnight stability test — RUNNING

Started **2026-09-16 20:41:57 SAST**, Python PID **6107** (confirm before signalling
in a later session; PIDs can be reused). 19 cycles x 39 clips = **741 events** at
30-second spacing. Expected finish ~**2026-09-17 02:52 SAST**. Hard timeout 7h.
Detached with nohup; no chat monitor is required for it to continue.

Command executed on Pi, from the isolated staging root:

```bash
nohup timeout --signal=TERM 7h /opt/smartfeeder/.venv/bin/python -u \
  deploy/soak_replay.py --config ml/reports/replay-fixed.yaml \
  --clips soak-clips --out ml/reports/soak-overnight \
  --cycles 19 --interval 30 \
  > ml/reports/soak-overnight.log 2>&1 < /dev/null &
```

- One pipeline reused for the whole run; no camera/GPIO opened by replay.
- Live `birdcam-capture` stays active. Mock replays bypass PIR admission/cooldown
  and camera timing; this soak does NOT validate those hardware paths.
- JSONL flushed after every event; status JSON replaced atomically. Logs decisions,
  action/identification rubrics, latency, RSS, temperature, clock and raw flags.
- Stops on runtime errors, current low-four-bit board warnings, RSS >512 MiB,
  or timeout. Historical sticky flag 0x80000 alone does not stop the test.
- Known classification errors are logged, not treated as runtime crashes.
- Repeating 39 clips 19 times does NOT supply 741 independent accuracy samples.
- Current verification confirmed first event processed and live service active.

Check from laptop:

```bash
ssh grilledbiscuits@192.168.0.224 \
  'cat ~/birdcam-next/desk-test/ml/reports/soak-overnight/status.json'
ssh grilledbiscuits@192.168.0.224 \
  'tail -5 ~/birdcam-next/desk-test/ml/reports/soak-overnight.log'
```

To stop intentionally: verify PID/command using status and `ps`, then send SIGTERM
(e.g. `kill -TERM 6107` while that PID is still the documented Python process).
The runner handles SIGTERM and writes state `stopped`.

Local snapshots of remote reports: `ml/reports/desk-20260916/pi/`.
Refresh after completion (safe: copies only from test report directory):

```bash
rsync -a grilledbiscuits@192.168.0.224:birdcam-next/desk-test/ml/reports/ \
  ml/reports/desk-20260916/pi/
```

### Next-session priorities

1. Read overnight status, collect full JSONL, summarize memory trend after warmup,
   temperature/current flags, errors, outcome stability and final live-service state.
2. Inspect the five wrong-action clips above visually against their source frames;
   check crop/scene changes and labels before changing thresholds. Do NOT retune
   to make these repeatedly examined development clips look perfect.
3. Validate on separate visits/views, including unknown animals and more hands.
   The current 39-clip set has no squirrels and no field-test Drongo clips; it is
   not a broad open-set test. Historical session/taxon split background overlap
   remains a methodology limitation.
4. Require a genuinely empty reference from the installed field camera, and
   independent empty/occupied sequences spanning lighting changes. This fixes
   foreground absorption, not the whole empty-scene problem; subjects outside the
   scored port region remain a limitation.
5. Do not run swap_in.sh yet. Its rollback/log checks still need controlled failure
   testing; broad passwordless sudo from the previous session remains unchanged.
   No permission files were edited by Codex.
6. Modern calibration metadata carries graph/site hashes, but existing startup
   pairing primarily checks checkpoint identity. Fully enforce graph, prior,
   suppression and per-class threshold provenance before a production swap.

Final local verification before this handover: 292 tests passing, Ruff checks
passing. Deployment shell syntax passed earlier; neither swap/rollback executed.


## Overnight completion — verified 2026-09-17

- Finished normally at 02:52:27 SAST, elapsed 6.175 hours, 741 events / 19 cycles
  of 39 distinct clips. Zero runtime errors. No label/outcome/gate variations
  for a given clip across cycles.
- 646/741 correct actions = the same 34/39 distinct clips passing every time;
  repeated trials are stability evidence, not independent accuracy samples.
- Same five wrong-action clips listed above remain: one empty published, two
  Amethyst visits discarded at family fallback, two Southern Double-collared
  visits discarded as other_animal. No thresholds changed after seeing results.
- Sampled temperature 39.4–47.7 C. Every sampled throttle word was 0x80000;
  no current low-four-bit warning flags appeared. Historical flag unchanged.
- RSS peaked at 162.793 MiB. Cycle median rose from 157.352 MiB during initial
  warmup to 162.543 on cycle 1 and 162.793 on cycle 18: small subsequent growth
  (~0.25 MiB), no large memory accumulation in this six-hour test. This does not
  prove indefinite leak-free operation.
- Live birdcam-capture remained active, NRestarts=0, active since 2026-09-16
  19:55:37 SAST. Current board temperature at this check was 38.4 C.
- Complete event log and status retrieved with rsync to
  `ml/reports/desk-20260916/pi/soak-overnight/`; derived summary in `analysis.json`.
- No service/config/model changes made while checking completion. Next task is
  visual diagnosis of the five failures, followed by independent scene/visit
  validation. Camera/PIR timing was not part of this replay soak.
