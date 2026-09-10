# In-flight run

Scratch state for a long job currently executing. Delete when it finishes and
the results are folded into DECISIONS.md / ASSUMPTIONS.md.

## Field extraction from drive B (2026-09-10)

Started 11:12. 13 uncut sessions plus 8 cut folders; expected finish ~12:35.

```bash
uv run python -m birdcam.data.field --cut-fps 2 --uncut-fps 1 --extract-only
```

- **log**: `/tmp/claude-1000/-home-grilledbiscuits-Desktop-smartfeeder/49527019-82fa-442b-9792-e06d2f466dd7/scratchpad/extract.log`
- **throughput**: ~2.5x realtime decode; 296 min of footage total
- **output**: `ml/data/field/frames/<folder>/`, projected ~22,000 frames / ~500 MB
- Idempotent and resumable: re-running skips clips whose frames already exist
  and re-validates them, so an interrupted run is repaired by running it again.

### `ml/training/` is symlinks to a USB drive

`ml/training/*` are symlinks into `/media/grilledbiscuits/B`. **If drive B is
unmounted the symlinks dangle and extraction cannot resume** — remount it, or
re-point them. The drive is otherwise untouched; nothing was written to it.

`hands/` and `multibird/` are real directories holding symlinks to loose clips
that sit at the drive root rather than in a species folder.

### The August extraction is preserved, not deleted

`ml/data/field_2026-08/` holds the previous corpus (11,050 frames,
`predictions.npz`, `whiteeye_candidates.json`). The 2026-09 drive supersedes it:
those clips were hand-edited by the observer, where the August cuts contained
verified empty frames (~12%) and multi-bird frames. A27's numbers were computed
from the August set and still refer to it.

## BLOCKER: the checkpoint no longer loads

Adding the Fork-tailed Drongo took the taxon head from 62 classes to 64, so
`student_best.pt` fails `load_state_dict` and `birdcam.data.field` can only run
with `--extract-only`. This is expected, not a fault: a new species requires a
retrain. Until then:

- no prediction, no domain-gap analysis, no threshold refit
- `capture/` cannot classify either; its ONNX is older still (A26)

## What to do when extraction finishes

1. Mine empty-feeder negatives from the new uncut footage:
   ```bash
   uv run python -m birdcam.data.mine_negatives
   ```
   Verified 96% precision at the default 30th percentile (1 bird in 24 sampled
   frames). **Precision falls as `--empty-percentile` rises** — the module
   docstring records 83% and 67% for the two approaches that came before.
2. Retrain. The corpus roughly doubles, gains a species, and gains the first
   negative-class examples the project has ever had.
3. Only then: refit thresholds, rebuild the novelty bundle, re-export ONNX.

## Still missing from Tier A

Cape Sugarbird, Greater Double-collared Sunbird, Malachite Sunbird and
Orange-breasted Sunbird have **no field footage at all**. Every field number is
computed over Southern Double-collared and Amethyst Sunbird only.
