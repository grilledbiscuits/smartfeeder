# In flight — Pi deployment (updated 2026-09-17)

**Not deployed.** The Pi (`sunfeed`, 192.168.0.224, key auth; `.local` does not
resolve) still runs the **August 62-class FP32** build from `/opt/smartfeeder`.
It is on a desk, not at the feeder. The full history of the 2026-09-16 soak is
in commit d701154's version of this file.

## Candidate

- 37-class checkpoint, best epoch 13 (SHA `0311dc3c…`), INT8 with field calibration.
- Capture-parity calibration (`birdcam.eval.calibrate_capture`, INT8 logits, site
  prior, `empty_feeder` suppressed): **T = 1.3907, energy threshold −4.7639**.
  Candidate files, not yet in `ml/config/`:
  `ml/reports/desk-20260916/{operating_points.json,taxonomy.yaml,replay-fixed.yaml}`.
- Do not use `eval/thresholds.py` for deployment thresholds: it fits on raw logits.

## Evidence so far (development clips, not a holdout)

Isolated Pi replay soak, 2026-09-16 20:42 → 09-17 02:52: 741 events (19 × 39
clips), 0 runtime errors, identical decisions every cycle, RSS peak 163 MiB,
39–48 °C, no current throttle flags. **34/39 correct actions**, same five wrong:

| Clip | Result | Diagnosis (2026-09-17) |
|---|---|---|
| empty4 | published as double-collared (genus) | Gate used a single fixed reference; lighting drift. Addressed by the idle-snapshot gate below. |
| Amethyst 3, 4 | family fallback, discarded | Female Amethyst at 0.94–0.96 family confidence; labels are right. **Family fallbacks of target families now publish.** |
| Double-collared 21, 3 (session 20260828_155630) | `other_animal`, discarded | Portrait phone video pillarboxed to 455×256 (68% black). The session's only training frames are 8 pillarboxed **hand** frames; its birds are all in test. A letterbox + scene shortcut. The Pi camera never produces pillarboxed frames, so these two clips are not representative of deployment. |

Pillarboxed sessions (portrait video in landscape frames): 20260828_155630,
20260828_163135, 20260901_161921, 20260901_173613. Candidate fix for the next
retrain: crop the black bars at extraction. Not done.

## Changes since the soak (2026-09-21, uncommitted until reviewed)

1. **Family-level answers are kept** when the family contains a Tier A species
   (`nectariniidae_indet`, `promeropidae_indet`). The guild fallback still
   discards. Replay scoring counts a family answer containing the truth as correct.
2. **Empty gate rebuilt: `IdleBackgroundGate`.** Background is the median of the
   last 6 snapshots taken every 300 s once the PIR has been quiet for 120 s;
   triggered clips never update it. Fails open when the newest snapshot is older
   than 900 s, when fewer than 3 exist, when the port is not found, or when frame
   shapes differ (now logged once instead of silent). The recorder stops the
   camera after each snapshot. `empty_gate.reference` is an optional seed for
   replay only; replay configs set `max_age_seconds: null`.
3. **Pillarboxed frames quarantined** (`birdcam.data.letterbox`, detected per
   frame, cached in `ml/data/letterboxed.json`): 2844 of 9139 field frames.
   Four sessions hold BOTH kinds, so a per-session rule is wrong.
   - Calibration refitted without them (`ml/reports/clean-20260921/`):
     T 1.3907 -> **1.4409**, energy threshold -4.7639 -> **-4.9434**. All six
     Tier A classes still reach 0.8 validation precision; Amethyst's threshold
     drops 0.54 -> 0.43.
   - Replay sets rebuilt (`deploy/build_field_replays.py --empty-runs 4`):
     20 visit clips in `clean-20260921/field-replay`, plus
     `clean-20260921/gate-20260827_152604` (1 empty clip, 5 Double-collared,
     1 Cape Bulbul, and a same-session `reference.jpg` seed — the seed's own run
     is excluded). Only one session has enough consecutive empty test frames, so
     gate evidence from replay is thin; the installed camera must supply the rest.
   - `ml/reports/framing/framing_accuracy.md` reports field accuracy split by
     framing, for the checkpoint and for the INT8 capture path.

### Local replay results, 2026-09-21 (laptop, INT8, new calibration)

- `gate-20260827_152604`: **7/7** — the empty clip was discarded, all six visits
  identified exactly.
- `field-replay` (20 clips): **18/20** by the identification rubric. Both
  remaining misses are worth knowing:
  - `16_..._hand3` -> `unknown` -> discarded. The action is right; the rubric
    counts it wrong because the truth label is `other_animal`.
  - `03_..._amethyst4` -> **`cinnyris_chalybeus` at 0.77, published**. A female
    Amethyst published under the wrong species. It was `nectariniidae_indet`
    (discarded) before this calibration.
- Amethyst calls became less specific overall: of five visits, 1 species-correct,
  3 family, 1 wrong species, against 3 species-correct and 2 discarded before.
  All five now publish, which is the point of the family policy, but the
  species-level labels on female Amethyst got worse, not better. Five clips is
  too few to retune on; see the retrain option below.

## Next, in order

1. Re-sync code to the Pi staging root (`~/birdcam-next/desk-test`) with the
   rebuilt replay sets and the `clean-20260921` calibration, and rerun there
   (the laptop run above is not a latency or hardware test).
2. **On the Pi, with the camera at the feeder:** confirm `capture_array` frames
   are RGB (the gate finds the port by redness; BGR would silently disable it),
   and measure snapshot-vs-decoded-clip departure on an empty port to check the
   12.0 threshold survives the preview-stream/H.264 difference.
3. Build the deploy config (INT8 graph, candidate operating points and taxonomy,
   `empty_gate` defaults), then deploy with `deploy/swap_in.sh ~/birdcam-next`
   inside tmux/nohup. `swap_in.sh`/`rollback.sh` have never been run.
4. Soak at the feeder; check errors, memory, `vcgencmd get_throttled`.
5. Narrow passwordless sudo on the Pi (currently ALL commands).

## Retrain option (not done)

Cropping the bars and retraining would give back 2844 correctly proportioned
frames, 1539 of them training frames, including 382 Amethyst. That is the one
change likely to improve female Amethyst at species level, which is now the
weakest real case. Costs a training run plus re-quantising and re-calibrating,
and the crop recovers geometry but not detail: the picture is 145px wide before
being scaled back up. The original footage is on Proton Drive; the extracted
frames only hold the barred version.

Quantisation calibration also used barred frames. INT8 is fine on
deployment-framed validation (0.933 against 0.927 for FP32) and collapses only
on barred frames, so re-quantising is not urgent — but do it in the same pass
as any retrain.
