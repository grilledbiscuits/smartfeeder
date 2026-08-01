# Decisions

Non-obvious choices and their reasoning. Newest phase last.

---

## Phase 1 — scaffolding

### D1. Dedicated git repository for `smartfeeder/`

A git repository already existed at `/home/grilledbiscuits/` — the entire home
directory, with zero commits, almost certainly an accidental `git init`.
Committing this project from within it would have staged `.ssh/`, `.gnupg/`,
`.claude.json`, browser profiles and Thunderbird mail.

`git init` was run inside `smartfeeder/` instead; the inner repository takes
precedence for all paths beneath it. The home repository was left untouched — it
is not ours to delete — but it is a standing hazard and is flagged in
`ASSUMPTIONS.md`.

### D2. Taxon IDs resolved at runtime, cache committed

`config/species.yaml` holds names only. GBIF `usageKey` and iNaturalist
`taxon_id` are resolved by `data/taxa.py` and cached to
`config/taxon_cache.json`.

The cache **is** committed, unlike everything else generated. It is small, and
pinning it makes runs reproducible when the GBIF backbone is revised — which it
is, regularly. Deleting the file forces a fresh resolution.

### D3. `male_breeding` vs `male_eclipse` uses a masked partial-label loss

Measured against the live iNaturalist controlled-terms API on 2026-07-31: the
Sex term (`id=9`) has exactly three values — Male=11, Female=10, Cannot Be
Determined=20. Life Stage (`id=1`) offers Adult=2 and Juvenile=8. **No public
source annotates eclipse plumage.** GBIF's `sex` field is likewise binary.

Three options were considered:

1. Map every annotated male to `male_breeding`. Rejected: it fabricates labels,
   and it matters — male *Nectarinia famosa* eclipse plumage is strikingly
   different from breeding, so the model would be actively taught that eclipse
   males are breeding males.
2. Drop `male_eclipse` from the schema. Rejected: it forecloses the distinction
   permanently and would require a schema migration to restore.
3. **Chosen:** keep the six-class schema, and train annotated males with a
   masked loss — `-log(p_male_breeding + p_male_eclipse)`. The sample
   contributes gradient to "this is a male" but not to which plumage state.

Nothing is invented, the output space never changes, and hand-labelled eclipse
images slot in later as exact labels with no migration. Declared in
`taxonomy.yaml` under `partial_label_groups`.

### D4. `nectarivore_indet` guild node above the family fallbacks

Verified against GBIF 2026-07-31: *Promerops cafer* is **Promeropidae**, not
Nectariniidae. The briefed `nectariniidae_indet` family fallback therefore does
not cover the Cape Sugarbird — a Tier A target.

A functional `guild` node spanning Nectariniidae + Promeropidae was added above
the family level. It is explicitly *not* a taxonomic rank. The justification is
that the downstream capture application's real question is "is this a nectar
feeder visitor worth recording?", which spans the taxonomic split, and the
rollup should be able to answer that question when family itself is uncertain.

### D5. Genus fallback classes only for multi-species genera

A genus holding one species in the label space gives the model nothing to be
uncertain *between*: its genus probability merely duplicates the species
probability, and the class can never accumulate training examples of its own.
Such a class is untrainable and dead.

Single-species genera (*Nectarinia*, *Anthobaphes*, *Hedydipna*, *Anthreptes*,
and nine Tier C genera) roll straight up to family. Eight genera qualify for a
genus fallback: Cinnyris (9 species), Pycnonotus (3), and six with 2.

Both directions are enforced in `Config.validate()` — a dead class is an error,
and so is a multi-species genus *without* a fallback, since dropping
`cinnyris_indet` would remove the single most important uncertainty this project
needs to express.

### D6. Validation is fatal, not advisory

`Config.validate()` raises rather than warns. Every check corresponds to a
mistake that would otherwise surface as a model that trains happily and means
nothing: a species with no rollup parent, a sex annotation mapped to a
nonexistent class, rollup thresholds ordered backwards, split fractions that do
not sum to 1.

This paid for itself immediately — the first run rejected the initial
`taxonomy.yaml` for nine Tier C genera with no rollup parent, which had been
written by hand and looked fine.

### D7. iNaturalist `large` photo size, not `original`

Originals run to 4000px and several MB. Everything is downsized to 256px short
side during preprocessing, and the development machine has 28 GB free. `large`
is capped at 1024px — ample headroom over 256 — and cuts download volume and
disk footprint by roughly an order of magnitude.

### D8. GBIF fetcher excludes the iNaturalist mirror dataset

Measured 2026-07-31: GBIF returns 6129 records for *Cinnyris chalybeus* in ZA
against iNaturalist's 6203, and the sampled record came from dataset
`50c9509d-22c7-4a22-a47d-8c48425ef4a7` — the iNaturalist research-grade export.
GBIF is therefore ~95% redundant for these taxa.

The iNat fetcher is preferred for those records because it also carries the
annotations. GBIF's value is the *other* datasets, so the mirror dataset key is
excluded by default in `train.yaml`.

Consequently the **async download endpoint is not used**: at the residual
volumes the search API is sufficient, which avoids the free-account credential
requirement entirely. If volumes grow past a few tens of thousands this should
be revisited, and the download DOI recorded.

### D9. Tier B rare species fetched globally, not ZA-restricted

Measured 2026-07-31: *Cinnyris manoensis* and *Cinnyris cupreus* have **zero**
research-grade South African records; *Anthreptes reichenowi* has 11 in ZA
against 33 globally, *Cinnyris neergaardi* 29 against 32. A ZA-restricted fetch
would return nothing for two of them.

`fetch_scope` is therefore per-tier in `species.yaml`: Tier A and C are ZA-only
(feeder-realistic), Tier B is global. Species still falling below
`min_species_images` are folded into their genus fallback class at
manifest-build time, and the fold is reported rather than silent.

### D10. Local fast-loop backbone is a mid-size CNN, with a Kaggle escape hatch

Every iNat-2021 checkpoint in `timm` is ViT-L/336 or ConvNeXt-L (verified
against the HuggingFace hub 2026-07-31). Both are far too slow for CPU embedding
extraction on 4 cores, and a transformer is forbidden for the student anyway
(Hailo).

`cache_embeddings.py` is architecture-agnostic by design. The local loop uses
`convnext_tiny.in12k_ft_in1k` and its separability verdicts are treated as a
**lower bound**. Caching with the ViT-L teacher on Kaggle and downloading the
`.npy` (20k × 1024 float32 ≈ 80 MB) gives the fast loop teacher-grade features
with no local GPU and no code change.

### D11. Separability verdicts require a minimum test-set size

Measured female counts on iNaturalist (ZA, research grade, 2026-07-31):
*C. chalybeus* 161, *C. afer* 108, *P. cafer* 59. After dedup and a grouped
70/15/15 split, the two *Cinnyris* classes yield roughly 16–24 test images each.

A merge recommendation derived from n=16 is not trustworthy — the 95% confidence
interval on an accuracy estimate at that sample size spans tens of percentage
points. `min_test_images_for_verdict: 50` is therefore enforced in
`taxonomy.yaml`: below it the fast loop reports **"insufficient data to decide"**
rather than a verdict, and every reported figure carries a confidence interval.

Merging *C. chalybeus* and *C. afer* females into `cinnyris_double_collared_indet`
remains a correct outcome if the data supports it. It is not correct to claim
it on the strength of 16 images.

---

## Vertical slice (Phases 2-5, narrow) — Tier A only

### D12. Annotated observations fetched before unannotated ones

Sex annotations, not photographs, are the scarce resource: 5-30% of
research-grade observations carry one. The iNat fetcher therefore runs a `sexed`
query variant (`term_id=9`) to exhaustion before falling back to `general`.

Consequence to remember when reading any metric from this corpus: **it is
deliberately not a representative sample of iNaturalist.** Females are massively
over-represented relative to the wild distribution. That is correct for training
the sex head and wrong for estimating real-world class priors, which must come
from the deployment site instead.

### D13. Split group key is (observer_id, scientific_name)

Grouping by `observation_id` alone stops bursts straddling but not photographer
style. Grouping by `observer_id` alone fixes style but spans species, which
breaks per-species stratification and can push a rare class wholly into one
split.

The pair does both: every group holds one species, so stratification works, and
since an observation has one observer and one species, no observation can
straddle. Both properties are asserted in `tests/test_data.py`.

### D14. Fast-loop head is converged, not undertrained

Full-batch gradient descent at 60 steps looked suspiciously few. Measured:
60 epochs -> 0.684, 200 -> 0.692, 600 -> 0.679, 1500 -> 0.677, 3000 -> 0.675.
It plateaus by ~200 and then mildly overfits. Cross-checked against an
independent solver (sklearn `LogisticRegression`, 0.690) and a 2-layer MLP head
(0.705). The linear-probe figure is real, not an artefact of the optimiser.
