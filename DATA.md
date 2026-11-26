# Project data

What is published, and how to look at it. Most links open in the browser.

## Browse the image records

**[Open the manifest in Datasette Lite](https://lite.datasette.io/?url=https://raw.githubusercontent.com/grilledbiscuits/smartfeeder/master/ml/data/manifest.sqlite)**:
search, filter and run SQL on [`manifest.sqlite`](ml/data/manifest.sqlite) with nothing to install.
It downloads about 26 MB, so give it a moment. Datasette Lite is a third-party viewer that runs in your browser.

Each row is one iNaturalist image: source links, licence, credit, species, date, location and train/validation/test split.
The images themselves are not in this repository; follow `page_url` to see one at its source.

Or open it with Python's standard library, from the repository root:

```bash
python3 -c "import sqlite3; [print(r) for r in sqlite3.connect('ml/data/manifest.sqlite').execute('select license, count(*) from images group by license')]"
```

## Results

### Classifier on the held-out test split

As reported in the thesis for the deployed INT8 model with the Pi's preprocessing:

| | All (5,930) | Web images (4,696) | Feeder frames (1,234) |
|---|---|---|---|
| Top-1 accuracy | 72.4% | 69.0% | 85.0% |
| Top-3 accuracy | 87.4% | 85.3% | 95.3% |

- The test split was also used to choose the INT8 calibration, so these figures may be slightly optimistic.
- The per-image predictions behind this table are not published. The closest published file is
  [`quantisation.json`](ml/reports/quantisation.json): FP32 76.2% and INT8 73.1% on the same split, using training preprocessing.

### Result files

| What | File |
|---|---|
| Training history, per epoch | [`training_history.json`](ml/reports/training_history.json) |
| FP32 against INT8, per class | [`quantisation.json`](ml/reports/quantisation.json) |
| Inference speed on the Pi 4B | [`bench_pi_4b.json`](ml/reports/bench_pi_4b.json) |
| Per-species publish thresholds (21 September refit) | [`operating_points.json`](ml/reports/clean-20260921/operating_points.json) |
| Accuracy on upright and padded video frames | [`framing_accuracy.md`](ml/reports/framing/framing_accuracy.md) |
| Desk and Pi replay tests, 16 September | [`desk-20260916/`](ml/reports/desk-20260916) |
| Early experiments: backbone and unknown-animal rejection | [`backbone_sweep.json`](ml/reports/backbone_sweep.json), [`open_set.json`](ml/reports/open_set.json) |

[`operating_points.json`](ml/reports/operating_points.json) at the top of `ml/reports/` is an earlier version.
[`operating_points_finetuned.json`](ml/reports/operating_points_finetuned.json) was fitted on an older model
(`a71e95cca471`) than the one published here (`0311dc3cf26a`).

### Field and hardware measurements

| Test | Headline | Details |
|---|---|---|
| Field commissioning, 6 and 9 October | Method and findings; only 3 wild visits observed | [log](deploy/field-log-20261009.md), [observations](deploy/field-log-20261009-observations.csv) |
| 12-hour replay soak | 1,440 events, 0 runtime errors, 1,368 correct identifications | [soak](deploy/soak-20261012) |
| Battery, idle | at least 17 h 1 min | [13 October](deploy/battery-20261013), [repeat: 17 h 13 min](deploy/battery-idle-repeat-20261016) |
| Battery, replaying saved clips | at least 18 h 31 min | [14 October](deploy/battery-inference-20261014) |
| Battery, recording and classifying every 30 s | at least 14 h 11 min | [15 October](deploy/battery-camera-inference-20261015) |
| Pi health telemetry | temperature, throttling, memory | [12 October](deploy/telemetry-20261012) |

The field log records that the camera was focused at 1 m, not at the perch, until 9 October 2026.
Feeder frames and thresholds from before that date come from blurred input.

## Model

| File | What it is |
|---|---|
| [`birdcam_student_int8.onnx`](ml/data/export/birdcam_student_int8.onnx) | Deployed model, 7.5 MB |
| [`birdcam_student_int8.json`](ml/data/export/birdcam_student_int8.json) | Class order, input format and checkpoint hash |

## Images per species

Downloaded images only (duplicates and failed downloads excluded). "No split" images are not in the
train, validation or test splits. Feeder video frames are not part of this manifest.

| Group | Species | Train | Validation | Test | No split | Total |
|---|---|---|---|---|---|---|
| A (targets) | Amethyst Sunbird | 1,209 | 259 | 257 | 0 | 1,725 |
| A (targets) | Malachite Sunbird | 1,105 | 236 | 235 | 0 | 1,576 |
| A (targets) | Greater Double-collared Sunbird | 1,093 | 235 | 234 | 0 | 1,562 |
| A (targets) | Orange-breasted Sunbird | 1,070 | 229 | 229 | 0 | 1,528 |
| A (targets) | Southern Double-collared Sunbird | 1,033 | 222 | 220 | 0 | 1,475 |
| A (targets) | Cape Sugarbird | 1,015 | 218 | 217 | 0 | 1,450 |
| B (wider nectar feeders) | Collared Sunbird | 1,025 | 220 | 219 | 0 | 1,464 |
| B (wider nectar feeders) | Marico Sunbird | 997 | 203 | 195 | 0 | 1,395 |
| B (wider nectar feeders) | Scarlet-chested Sunbird | 942 | 203 | 202 | 0 | 1,347 |
| B (wider nectar feeders) | White-bellied Sunbird | 934 | 200 | 200 | 0 | 1,334 |
| B (wider nectar feeders) | Dusky Sunbird | 861 | 185 | 183 | 0 | 1,229 |
| B (wider nectar feeders) | Olive Sunbird | 764 | 167 | 160 | 0 | 1,091 |
| B (wider nectar feeders) | Purple-banded Sunbird | 743 | 156 | 163 | 0 | 1,062 |
| B (wider nectar feeders) | Grey Sunbird | 385 | 72 | 72 | 0 | 529 |
| B (wider nectar feeders) | Copper Sunbird | 344 | 73 | 72 | 0 | 489 |
| B (wider nectar feeders) | Gurney's Sugarbird | 226 | 44 | 44 | 0 | 314 |
| B (wider nectar feeders) | Miombo Double-collared Sunbird | 150 | 42 | 25 | 0 | 217 |
| B (wider nectar feeders) | Plain-backed Sunbird | 32 | 7 | 6 | 0 | 45 |
| B (wider nectar feeders) | Neergaard's Sunbird | 27 | 6 | 5 | 0 | 38 |
| C (look-alikes) | Fork-tailed Drongo | 1,194 | 247 | 246 | 0 | 1,687 |
| C (look-alikes) | Fiscal Flycatcher | 1,132 | 243 | 242 | 0 | 1,617 |
| C (look-alikes) | House Sparrow | 465 | 101 | 99 | 0 | 665 |
| C (look-alikes) | Cape Canary | 459 | 99 | 97 | 0 | 655 |
| C (look-alikes) | Southern Fiscal | 451 | 97 | 93 | 0 | 641 |
| C (look-alikes) | Cape Weaver | 414 | 88 | 88 | 0 | 590 |
| C (look-alikes) | Cape Sparrow | 411 | 88 | 87 | 0 | 586 |
| C (look-alikes) | Red-faced Mousebird | 407 | 88 | 86 | 0 | 581 |
| C (look-alikes) | Common Starling | 425 | 75 | 74 | 0 | 574 |
| C (look-alikes) | Cape Bulbul | 400 | 87 | 85 | 0 | 572 |
| C (look-alikes) | Southern Masked Weaver | 397 | 85 | 85 | 0 | 567 |
| C (look-alikes) | Karoo Prinia | 390 | 80 | 81 | 0 | 551 |
| C (look-alikes) | Cape White-eye | 405 | 63 | 62 | 0 | 530 |
| C (look-alikes) | African Red-eyed Bulbul | 350 | 72 | 72 | 0 | 494 |
| C (look-alikes) | Orange River White-eye | 340 | 73 | 73 | 0 | 486 |
| C (look-alikes) | Cape Glossy Starling | 337 | 73 | 71 | 0 | 481 |
| C (look-alikes) | Red-winged Starling | 314 | 68 | 68 | 0 | 450 |
| C (look-alikes) | Speckled Mousebird | 283 | 60 | 60 | 0 | 403 |
| Not birds (open-set tests) | Citrus Swallowtail | 0 | 0 | 0 | 354 | 354 |
| Not birds (open-set tests) | Four-striped Grass Mouse | 0 | 0 | 0 | 347 | 347 |
| Not birds (open-set tests) | African Carpenter Bee | 0 | 0 | 0 | 281 | 281 |
| Not birds (open-set tests) | Domestic Cat | 0 | 0 | 0 | 247 | 247 |
| Not birds (open-set tests) | Black Rat | 0 | 0 | 0 | 238 | 238 |
| Not birds (open-set tests) | Cape Skink | 0 | 0 | 0 | 227 | 227 |
| Not birds (open-set tests) | Eastern Grey Squirrel | 0 | 0 | 0 | 213 | 213 |
| Not birds (open-set tests) | Chacma Baboon | 0 | 0 | 0 | 213 | 213 |
| Not birds (open-set tests) | Western Honey Bee | 0 | 0 | 0 | 202 | 202 |
| Not birds (open-set tests) | Southern Rock Agama | 0 | 0 | 0 | 164 | 164 |
| **Total** | | **22,529** | **4,764** | **4,707** | **2,486** | **34,486** |

## Not published

Source images, feeder video and frames, embeddings, training checkpoints, other model exports,
video clips from the replay tests, and the older HTML metrics report.

## Licences

The image records come from iNaturalist observations licensed CC0, CC BY or CC BY-NC (most are CC BY-NC).
Reuse needs the credit in `attribution_text`, and non-commercial licences also apply to that use.
Treat the trained model as non-commercial. This repository does not yet have a licence of its own.
