# In-flight run

Scratch state for work currently in progress. Delete when it finishes and the
results are folded into DECISIONS.md / ASSUMPTIONS.md.

## Drive B can be unmounted (2026-09-10)

Extraction finished at 12:07. Everything the training path needs is on local
disk:

- `ml/data/field/frames/` — 22,213 extracted frames, 513 MB, **real files, no
  symlinks**. Verified: 0 of 9,139 field training records resolve onto `/media`.
- `ml/data/field/frames.json` — the frame index.
- `ml/data/field/empty_candidates.json` — the mined empty-feeder set.

`ml/training/*` are still symlinks into `/media/grilledbiscuits/B` and will
dangle once the drive is out. That only matters if you want to **re-extract**
from the original video — remount the drive, or re-point them. Nothing was ever
written to the drive.

Frames are the reproducible artefact here, not the video. The video is the
archive copy.

## Ready to retrain

```bash
uv run python -m birdcam.train.train_full --estimate     # ETA first
uv run python -m birdcam.train.train_full
```

Measured on this laptop: 22.7 min/epoch, ~5.7 h for 15 epochs at
`freeze_blocks=4`. Checkpointed every epoch; `--resume` continues.

**Corpus: 41,056** = 31,917 web (iNaturalist) + 9,139 field.

All 37 taxon outputs have training data. Zero empty classes.

| field class | frames |
|---|---|
| Southern Double-collared Sunbird | 4,499 |
| Cape White-eye | 1,358 |
| Amethyst Sunbird | 1,220 |
| Fork-tailed Drongo | 1,093 |
| Empty feeder | 476 |
| Cape Bulbul | 450 |
| Other animal (hands) | 43 |

Sex head, after the two labelling passes: 6,923 female, 10,226 male, 588
juvenile, 11,849 not applicable, 11,518 indeterminate (all from web sources
that genuinely carry no annotation), 35 unsupervised. **No field frame claims
`indeterminate`.**

The 62-vs-64 checkpoint blocker is **gone**: the drongo now has 1,686 fetched
images of its own, so the new head is trained rather than warm-started, and no
class-index remapping is needed. `student_best.pt` is superseded, not reused.

### After the run

1. Refit per-class thresholds.
2. Rebuild the novelty bundle (`ml/data/export/novelty_knn.npz`).
3. Re-export ONNX — A26 records the on-disk export as the pre-fine-tune model.
4. Report web and field accuracy **separately**. A single number hides the thing
   worth knowing: whether the field frames helped in the field.

### The one number to move first

`train.field_weight` in `config/train.yaml` is **0.3**, and it is a placeholder
— chosen, not measured. 9,139 field frames come from about thirty recording
sessions; 21,449 web images come from thousands of photographers. If field
performance disappoints, this is the first dial.

## Still missing field footage entirely

Cape Sugarbird, Greater Double-collared Sunbird, Malachite Sunbird,
Orange-breasted Sunbird. All four are Tier A.

## Open, not blocking

- `multibird/` (75 frames) is quarantined: two birds in frame, no single taxon
  is true of it. Needs per-frame labels to be usable.
- Tier B (13 species) has no images by design — D9 folds them into genus
  fallbacks — but that means 13 head outputs can never receive a positive
  example. Worth revisiting whether they should be classes at all.
