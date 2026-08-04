# In-flight run

Scratch state for a long job currently executing. Delete when it finishes and
the results are folded into DECISIONS.md / ASSUMPTIONS.md.

## Full fine-tune (Phase 6)

Started 2026-08-04 18:04. Expected finish ~00:15 (≈6.2 hours).

```bash
uv run python -m birdcam.train.train_full \
    --freeze-blocks 0 --epochs 20 --batch-size 8 --accum 4
```

- **log**: `/tmp/claude-1000/-home-grilledbiscuits-Desktop-smartfeeder/9a2825fa-767a-49ff-943d-a028433ea203/scratchpad/finetune.log`
- **throughput**: ~11.5 img/s, 1,597 steps/epoch, ~18.5 min/epoch
- **peak memory**: 1,715 MB RSS against ~2,450 MB available — roughly 700 MB of
  headroom, so a heavy browser could push it over
- **checkpoints**: `data/checkpoints/student_last.pt` and `student_best.pt`,
  written every epoch (best selected on Tier A recall)

### Check progress

```bash
uv run python -m birdcam.train.train_full --history
```

Reads the checkpoint, so it works while the run is in flight. This run predates
the per-epoch `training_history.json` fix, so that file will only appear at the
end; `--history` is the way to see partial results.

### If it dies

Four OOM kills have already happened on this machine. A kill costs at most one
epoch:

```bash
uv run python -m birdcam.train.train_full \
    --freeze-blocks 0 --epochs 20 --batch-size 8 --accum 4 --resume
```

### What to do when it finishes

The fine-tuned checkpoint changes the inputs to everything downstream, so
re-run the analysis against it:

1. `uv run python -m birdcam.eval.thresholds --target-precision 0.80 --write-config`
   — fresh per-class operating points
2. `uv run python -m birdcam.eval.report` — the full picture

**The numbers to watch**, both from ASSUMPTIONS.md A20:

- **22.9%** of genuine bird visits currently clear a threshold. This is the
  binding constraint on the whole project. **See the correction below — the
  honest figure is ~41%.**
- ***Cinnyris chalybeus*** — the most likely visitor at Rondebosch, 2,858
  observations within 25 km — is the worst class, unable to reach 80% precision
  at any threshold and firing on 22% of its frames at 0.82.

If fine-tuning does not move those, the ceiling is not the head or the
thresholds, and the next lever is data rather than modelling.

## The 22.9% problem — deferred work

User asked on 2026-08-04 to park this until the fine-tune lands, then come back
to it. This section is the reminder.

### Correction to A20, not yet applied

`false_trigger_curve` counts triggers over **every** in-distribution test visit,
but `frame_triggers` only checks **Tier A** class thresholds, so Tier C visits
cannot fire by construction. Measured split of the 534 test visits (≥2 frames):

| tier | visits | share |
|---|---|---|
| A | 300 | 56.2% |
| C | 234 | 43.8% |

The metric's ceiling is therefore 56.2%, not 100%. Since only Tier A can fire,
0.229 × 534 = 122 fired visits, all Tier A, giving **≈41% of Tier A visits** —
not 22.9% of bird visits. Still poor, but A20 as written overstates it.

**A20 has not been edited yet.** Do it as part of the post-run re-measurement so
before/after use one definition.

### Why ~41% is still low — three mechanisms

1. **The novelty gate is suppressing real birds.** Per-class frame recalls
   weighted by actual test-visit counts give ~53.6% expected frame recall.
   Visit-level fires if *any* frame fires, so it should exceed that; it is
   lower. At 15% FAR, 15% of in-distribution frames are flagged unknown by
   definition, and the majority-unknown rule then kills whole visits. The
   open-set failsafe and the classifier are fighting each other, and the 0.4%
   OOD trigger rate is being paid for in missed birds.
2. **Frames within a visit are correlated** — same bird, pose, lighting,
   seconds apart. "Any frame fires" gives much less lift than independence
   would suggest.
3. **Genuinely poor confidence on hard classes.** Tier A visits are spread
   evenly (each species 13–22% of the 300), so *C. chalybeus* is not a
   rare-class artefact.

### Candidate solutions, roughly by expected value

- **Re-tune novelty FAR jointly with the class thresholds.** They were fitted
  independently. 15% FAR over-delivers on OOD (0.4%) while costing real birds.
  Cheapest lever, no retraining, likely the biggest single win.
- **Fix the visit metric**: Tier A denominator, and decide whether Tier C should
  trigger a capture at all. A rarer nectarivore is arguably *more* worth
  recording — currently they are silently unrecordable.
- **Teacher distillation** from an iNat-2021 ViT-L into the student
  (`models/distill.py`, still the Phase 1 stub). Most promising lever on
  *C. chalybeus*; realistically needs a CUDA box.
- **Label-noise triage** of the corpus — remove frames with no usable bird
  rather than relabelling them.
- **Error analysis** on *C. chalybeus* false positives and non-triggering
  visits: juveniles, females, backlit, edge-of-frame?

### Not yet done

- Ablation the brief asked for: class-balanced sampling vs focal loss, reporting
  which actually helps. Both are implemented and configurable; neither has been
  measured against the other.
- Quantisation sweep at a proper calibration size — capped at 64 images by RAM,
  so the 3.4–5.1pp INT8 penalty may be an artefact of the budget (A24b).
- Phase 8 capture application (`capture/`).
