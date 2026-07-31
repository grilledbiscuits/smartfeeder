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

## Configuration

Everything is driven by `config/`. There are no hardcoded species lists, taxon
IDs or paths in Python.

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

`data/` is gitignored. The manifest is the reproducible artefact, not the pixels.
