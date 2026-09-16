# In flight — Pi 4B deployment (paused 2026-09-16 ~20:10, session limit)

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
