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
  binding constraint on the whole project.
- ***Cinnyris chalybeus*** — the most likely visitor at Rondebosch, 2,858
  observations within 25 km — is the worst class, unable to reach 80% precision
  at any threshold and firing on 22% of its frames at 0.82.

If fine-tuning does not move those, the ceiling is not the head or the
thresholds, and the next lever is data rather than modelling.

### Not yet done

- Ablation the brief asked for: class-balanced sampling vs focal loss, reporting
  which actually helps. Both are implemented and configurable; neither has been
  measured against the other.
- Phase 8 capture application (`capture/`).
