# birdcam — South African nectarivore feeder-cam classifier

Fine-grained bird identification for a nectar feeder in Cape Town, South Africa.
Two prediction heads over a shared backbone:

- **Head 1 — taxon**: species, with honest fallback to genus / family / guild when
  the model cannot be confident.
- **Head 2 — sex / plumage**: `male_breeding`, `male_eclipse`, `female`, `juvenile`,
  `indeterminate`, `not_applicable`.

Both outputs carry calibrated confidence, because a downstream capture application
uses them to decide whether to record video.

Deployment target: Raspberry Pi 4B, running fully on-device with CPU inference.

## View the data

**[DATA.md](DATA.md)** has the results, field and battery measurements, the model, and a
[browser view of all image records](https://lite.datasette.io/?url=https://raw.githubusercontent.com/grilledbiscuits/smartfeeder/master/ml/data/manifest.sqlite)
that needs nothing installed.

## Status

Built and deployed on a Raspberry Pi 4B at a feeder, with field tests in October 2026:
data pipeline, training, INT8 export, motion-triggered capture and a web dashboard.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies resolve `torch` and
`torchvision` from the **CPU-only** PyTorch index — the development machine has
no discrete GPU, and the CUDA wheels are ~2.5 GB of dead weight.

```bash
uv sync --extra export
```

Fetchers require a contact email in the User-Agent, and refuse to run without it:

```bash
export BIRDCAM_CONTACT='you@example.com'
```

Optional, both skipped gracefully when absent: `FLICKR_API_KEY`, `WANDB_API_KEY`.

Run the tests:

```bash
uv run pytest -q
```

## Repository layout

The main areas of this repo:

- **`ml/`** — the classifier: `src/birdcam/`, `config/`, `tests/`, `reports/`,
  the gitignored `data/` corpus, and the design docs (`DECISIONS.md`,
  `ASSUMPTIONS.md`, `RUNNING.md`). Paths below are relative to `ml/`.
- **`capture/`** — the on-device service that detects motion, records, classifies
  and publishes. See [its README](capture/README.md).
- **`deploy/`** — deployment scripts, services and measurement logs.
- **`web/`** — the dashboard, with its runtime state in `var/`. It imports
  nothing from `birdcam`; see the interface contract at the end of this file.
- **`cad/`** — parametric feeder cradle, Raspberry Pi housing, and power box
  designs. See [CAD designs](cad/README.md) for source files and printable
  exports. The Pi housing fits the deployment board, a Pi 4B.

## Configuration

Everything is driven by `ml/config/`. There are no hardcoded species lists,
taxon IDs or paths in Python.

- `config/species.yaml` — the three species tiers. **Names only**; GBIF
  `usageKey` and iNaturalist `taxon_id` are resolved at runtime and cached to
  `config/taxon_cache.json`.
- `config/taxonomy.yaml` — label hierarchy, rollup thresholds, sex/plumage
  annotation mapping, class-size policy.
- `config/train.yaml` — paths, fetch settings, preprocessing, backbones,
  hyperparameters, augmentation, export settings.

`birdcam.config.Config.validate()` cross-checks these files against each other on
load and raises rather than warning. Every check corresponds to a mistake that
would otherwise surface as a quietly wrong model instead of an error — a species
with no rollup parent, a sex annotation mapped to a nonexistent class, rollup
thresholds ordered backwards.

## Known constraints

These shape the design and are not incidental. See `DECISIONS.md` for reasoning
and `ASSUMPTIONS.md` for what still needs verifying.

- **Development is CPU-only.** The fast iteration loop (Phase 5) runs on cached
  embeddings so head experiments finish in seconds. `train_full.py` must stay
  runnable on Kaggle unmodified.
- **Benchmark inference latency on the Pi 4B.** Development-laptop latency
  does not predict performance on its CPU. Accuracy and quantisation delta can
  be measured off-target; runtime must be checked on the board.
- **The current student backbone is a CNN.** Recheck its CPU latency on the
  Pi 4B before committing to the model; the earlier Hailo compiler constraint
  no longer applies.
- **`male_breeding` vs `male_eclipse` cannot be sourced automatically.** No
  public API annotates plumage state. Annotated males are trained with a masked
  partial-label loss rather than being assigned a fabricated label.

## Licensing of fetched imagery

Only CC0, CC-BY and CC-BY-NC images are retained. Anything with no licence
stated, or any ND / all-rights-reserved variant, is excluded. CC-BY and CC-BY-NC
both require attribution on downstream use, so the manifest records
`license`, `rights_holder` and `attribution_text` for **every** image.

Most of `ml/data/` is gitignored. The manifest is published; the images are not.
See [DATA.md](DATA.md).

## Web dashboard

A minimal local dashboard for browsing feeder visits, in `web/`. It only
reads: it never runs the classifier and doesn't import anything from
`birdcam`, so it can be stopped, restarted, or crashed without affecting the
feeder.

**Interface contract:** the feeder/inference process saves an image and/or
video under `var/media/{images,videos}/` and inserts one row per visit via
`web.db.add_visit()`. The dashboard just reads `var/feeder.db` and serves
whatever those filenames point to. The capture service calls it. Without a
feeder, `web/scripts/create_dummy_data.py` writes dummy visits instead.

Install the dashboard's dependencies (kept separate from the ML stack — this
doesn't pull in torch):

```bash
uv sync --extra web
```

Create some dummy visits to look at (regenerates `var/feeder.db` and sample
media each time; uses `ffmpeg` for placeholder video clips if it's on PATH):

```bash
uv run python -m web.scripts.create_dummy_data
```

Run the dashboard:

```bash
uv run python -m web.app
```

Open `http://localhost:5000` on the Pi itself, or `http://<pi-ip>:5000` from
another device on the same LAN — the dev server binds `0.0.0.0`.

Layout:

- `var/feeder.db` — SQLite `visits` table (WAL mode, so the feeder can write
  while the dashboard reads). Gitignored; regenerated locally.
- `var/media/images/`, `var/media/videos/` — the files `image_filename` /
  `video_filename` point to.
- `web/db.py` — the only module that touches SQL (`add_visit`,
  `get_recent_visits`).
- `web/app.py` — Flask routes: `/` renders recent visits, `/media/images/...`
  and `/media/videos/...` serve the corresponding files.

Not implemented yet, deliberately: auth, filtering, charts, live updates.
This is a v1 foundation, not the final dashboard.
