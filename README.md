# birdcam — South African nectarivore feeder-cam classifier

Fine-grained bird identification for a nectar feeder in Cape Town, South Africa.
Two prediction heads over a shared backbone:

- **Head 1 — taxon**: species, with honest fallback to genus / family / guild when
  the model cannot be confident.
- **Head 2 — sex / plumage**: `male_breeding`, `male_eclipse`, `female`, `juvenile`,
  `indeterminate`, `not_applicable`.

Both outputs carry calibrated confidence, because a downstream capture application
uses them to decide whether to record video.

Deployment target: Raspberry Pi 5 + Hailo-8L (AI HAT+), running fully on-device.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Repo scaffolding, config layer | **complete** |
| 2 | Data acquisition (iNat, GBIF, Wikimedia, Flickr) | not started |
| 3 | Preprocessing, dedup, grouped splits | not started |
| 4 | Backbone selection | not started |
| 5 | Fast iteration loop (cached embeddings) | not started |
| 6 | Full training | not started |
| 7 | Metrics and HTML report | not started |
| 8 | Export + Pi capture application | not started |

Modules for phases 2–8 exist as placeholders that raise `NotImplementedError`.
They deliberately do not return empty results: a fetcher that quietly returns
nothing is indistinguishable from a species that genuinely has no data, and that
distinction is the whole point of the data report.

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

Two independent projects share this repo:

- **`ml/`** — the classifier: `src/birdcam/`, `config/`, `tests/`, `reports/`,
  the gitignored `data/` corpus, and the design docs (`DECISIONS.md`,
  `ASSUMPTIONS.md`, `RUNNING.md`). Paths below are relative to `ml/`.
- **`web/`** — the dashboard, with its runtime state in `var/`. It imports
  nothing from `birdcam`; see the interface contract at the end of this file.

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
  embeddings so head experiments finish in seconds. Full fine-tunes run on
  Kaggle; `train_full.py` must stay runnable there unmodified.
- **Do not benchmark inference latency on the development laptop.** Latency
  measured off-target is meaningless for a Pi 5 + Hailo-8L. Accuracy and
  quantisation delta are the only performance metrics that belong in this repo.
- **The student backbone must remain a CNN.** The Hailo Dataflow Compiler has
  limited and awkward support for ViT and ConvNeXt blocks.
- **`male_breeding` vs `male_eclipse` cannot be sourced automatically.** No
  public API annotates plumage state. Annotated males are trained with a masked
  partial-label loss rather than being assigned a fabricated label.

## Licensing of fetched imagery

Only CC0, CC-BY and CC-BY-NC images are retained. Anything with no licence
stated, or any ND / all-rights-reserved variant, is excluded. CC-BY and CC-BY-NC
both require attribution on downstream use, so the manifest records
`license`, `rights_holder` and `attribution_text` for **every** image.

`ml/data/` is gitignored. The manifest is the reproducible artefact, not the
pixels.

## Web dashboard

A minimal local dashboard for browsing feeder visits, in `web/`. It only
reads: it never runs the classifier and doesn't import anything from
`birdcam`, so it can be stopped, restarted, or crashed without affecting the
feeder.

**Interface contract:** the feeder/inference process saves an image and/or
video under `var/media/{images,videos}/` and inserts one row per visit via
`web.db.add_visit()`. The dashboard just reads `var/feeder.db` and serves
whatever those filenames point to. That function isn't wired into the
classifier yet — for now, `web/scripts/create_dummy_data.py` plays the role
of the feeder process and writes realistic dummy visits.

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
