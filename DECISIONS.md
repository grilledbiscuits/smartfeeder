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

---

## Backbone selection and calibration (2026-08-01)

### D15. Student backbone chosen by measurement: `tf_efficientnetv2_b0`

Eight backbones, 9,318 Tier A images, linear probe on frozen features. Cost is
reported as params and MACs — architectural properties, identical on any
machine. Latency is deliberately not measured: this is an x86 laptop and the
target is a Pi 5 + Hailo-8L.

| backbone | MACs | params | taxon | female |
|---|---|---|---|---|
| convnext_tiny *(Hailo-ineligible, ceiling)* | 4.46G | 27.8M | 0.788 | 0.681 |
| **tf_efficientnetv2_b0** | **0.72G** | **5.9M** | **0.734** | **0.577** |
| efficientnet_b0 | 0.39G | 4.0M | 0.691 | 0.537 |
| mobilenetv3_large_100 | 0.21G | 4.2M | 0.690 | 0.495 |
| efficientnet_lite0 | 0.38G | 3.4M | 0.681 | 0.492 |
| mobilenetv4_conv_small | 0.18G | 2.5M | 0.680 | 0.469 |
| mobilenetv4_conv_medium | 0.83G | 8.4M | 0.670 | 0.478 |
| resnet50 | 4.09G | 23.5M | 0.664 | 0.511 |

`tf_efficientnetv2_b0` buys +4.4pp taxon and **+8.2pp female** over the previous
`mobilenetv3_large_100` default for 3.4x the MACs. Worth taking, because the
absolute cost is negligible: 0.72 GMACs is a rounding error for a Hailo-8L, and
the Pi-side bottleneck is the camera pipeline, not the accelerator — we classify
sampled frames, not every frame.

Two results worth keeping in view:

* **ResNet-50 scores *worse* than MobileNetV3 at 20x the compute.** More FLOPs is
  not more accuracy. This is exactly why the sweep exists rather than a guess.
* **`mobilenetv4_conv_medium` is worse than `_small`** despite 4.6x the MACs, on
  this data. Do not assume the bigger variant of a family wins.

### D16. Temperature scaling, fitted on val

The raw head is roughly 2x overconfident: ECE 0.183. A single scalar temperature
(T = 1.97) fitted on the **validation** split — never on test — reduces ECE to
**0.062**, and leaves argmax untouched, so accuracy is unchanged.

This matters more than it looks. Confidence gates the video-capture decision, so
a model that reports 0.95 on everything cannot drive a threshold at all. One
scalar is essentially free at inference and is applied outside the ONNX graph,
alongside rollup, so retuning never requires a Hailo recompile.

### D17. Rollup, thresholds and softmax stay outside the exported graph

The ONNX graph is: preprocessed tensor in, two logit tensors out. Softmax,
temperature, rollup and per-class thresholds are all applied by the caller.

Those values are tuned from precision-recall curves and change without
retraining. Baking them into the graph would force a Hailo recompile — a slow
step on a separate x86 toolchain — every time a threshold moved.

---

## Tier C and the open-set failsafe (2026-08-02)

### D18. The failsafe is fitted on target birds only, never on intruders

A softmax over the label space always returns one of its classes. Pointed at a
squirrel it reports a bird, often confidently, because nothing in training ever
offered "none of the above" as an option.

The detector is fitted on **in-distribution training features only**. It never
sees an out-of-distribution example, at fit time or calibration time.

That is the whole design. A detector trained on squirrels learns *squirrel*, and
the next intruder is a mongoose, a hand, a blown leaf, or rain on the lens.
Fitting only on what the target birds look like means anything sufficiently
unlike them is flagged — including things nobody enumerated. The OOD taxa in
`config/ood.yaml` exist to *measure* the detector, never to build it.

### D19. kNN scorer, full-width features, 1000 references

Measured over 18,146 target-bird images against 2,486 real photographs of
squirrels, mice, cats, rats, baboons, carpenter bees, honey bees, butterflies,
agamas and skinks, at a 5% false-alarm rate:

| scorer | AUROC | intruders caught | cost |
|---|---|---|---|
| **kNN** | **0.979** | **90.9%** | ~5 MB, 2.6 MFLOPs |
| energy | 0.927 | 68.1% | free (logits) |
| mahalanobis | 0.876 | 29.2% | ~66 KB, 1 matmul |
| max_softmax | 0.714 | 17.8% | free (logits) |

**A naive confidence threshold catches under a fifth.** That is the single most
important number here: the obvious solution does not work, because OOD inputs
frequently produce *confident* predictions rather than uncertain ones.

Two tuning results, both counter-intuitive:

* **Cutting references from 5000 to 1000 costs nothing** — AUROC 0.979 → 0.981.
  Take the cheap one.
* **PCA wrecks it.** 1280 dims → 90.9% caught; 256 → 57.3%; 128 → 47.5%; 64 →
  34.3%. The novelty signal lives in the low-variance directions PCA discards,
  which makes sense: the high-variance directions encode what separates bird
  species *from each other*, not what separates birds from squirrels.

Final cost: 2.6 MFLOPs per frame against the backbone's 1440. **0.18% overhead.**

### D20. `unknown` and `uncertain` are different labels

* `unknown` — "not a thing I know about" (squirrel, hand, rain)
* `uncertain` — "probably a bird, but I cannot pin it down"

They need different downstream behaviour. `uncertain` visits are the valuable
ones for the data flywheel: real birds worth a human look. `unknown` triggers are
mostly noise — though a sustained run of them usually means something changed at
the feeder, which is itself worth a notification.

### D21. Novelty runs BEFORE the taxonomic rollup, and short-circuits

Order is load-bearing. The rollup cannot express "not a bird" — given a squirrel
it will return the least-bad species, or roll up to `nectarivore_indet`, which is
worse than useless because it looks like a confident functional identification.
The novelty check therefore runs first and returns immediately. Asserted in
`tests/test_openset.py::test_novelty_short_circuits_before_any_species_is_named`.

### D22. A minority of unknown frames does not veto a visit

Track voting counts `unknown` frames but only calls the whole visit unknown if
they are the *majority*. A squirrel crossing behind a feeding sunbird should not
suppress the sunbird.

---

## Re-prioritisation: confidence and false triggers (2026-08-04)

### D23. Sex/plumage head demoted to auxiliary

`sex_weight` 0.4 → 0.1. Sex is a bonus, not a requirement, so the head is kept
only for its regularising effect on the shared trunk and for the label when it
happens to be right. It no longer influences backbone selection, thresholds, or
any headline metric.

Not removed outright: a small auxiliary loss on a correlated task is cheap
regularisation, and the six-class output space is already wired end to end.
Deleting it would be work now and work again later if it ever matters.

### D24. Hardware decision deferred; the export is board-neutral

The board is undecided between Pi 4B and Pi 5. The exported ONNX runs on both
and nothing in the model needs to change to switch, so the architecture stays as
it is and `reports/deployment.md` documents both paths instead of picking one.

Facts that will decide it, both verified 2026-08-04:

* **The AI HAT+ (Hailo-8L) connects over the Pi 5's PCIe port and does not work
  on a Pi 4.** Choosing the 4B means CPU-only inference.
* **The Pi 5 has no hardware H.264 encoder** — it was removed from the BCM2712;
  the Pi 4 retains one. So continuous encoding for the pre-roll buffer costs CPU
  on a Pi 5 and almost nothing on a Pi 4, and the Pi 4's encoder also exposes
  motion vectors that could replace frame differencing for free.

The Pi 5 is the better inference machine and the worse video machine; the Pi 4B
is the reverse. With sampled-frame classification and track voting, continuous
encoding is the larger constant load — which is not the obvious answer.

If the Pi 4B is chosen, revisit the backbone: MACs predict NPU cost well and ARM
CPU cost poorly (SE blocks stall the pipeline, swish is transcendental where
ReLU6 is a clamp). `efficientnet_lite0` exists for that case.

### D25. Per-class thresholds fitted for target precision, not accuracy

`per_class_thresholds` in `taxonomy.yaml` is now populated from real
precision-recall curves rather than a single global guess. Each Tier A species
gets the lowest threshold reaching the target precision on the **validation**
split; test precision and the recall paid for it are recorded inline.

Lowest rather than highest: among thresholds meeting the target we want the one
keeping the most recall.

### D26. Novelty false-alarm rate raised 0.05 → 0.15

Measured over 534 real bird visits and 540 OOD visits, reconstructed from
observation IDs so a "visit" is a genuine photo burst of one individual:

| novelty FAR | bird visits fire | OOD visits fire |
|---|---|---|
| 2% | 27.3% | 2.8% |
| 5% | 27.0% | 0.9% |
| **15%** | **22.9%** | **0.4%** |
| 20% | 22.1% | 0.0% |

Going from 5% to 15% removes three quarters of the remaining false triggers for
four percentage points of bird visits. Beyond 20% the OOD rate is already zero
and further sensitivity only loses birds.

**Visit level is the number that matters.** A visit is many frames and the track
vote decides, so frame-level rates systematically overstate the problem — 0.6%
of OOD *frames* versus 0.4% of OOD *visits* at the same setting.

---

## Range prior, calibration order, and compute trimming (2026-08-04)

### D27. Range prior built from real local observation density

`config/sites/rondebosch.yaml`, generated by `birdcam.data.range_prior` from the
iNaturalist `species_counts` endpoint within 25km of Rondebosch (376 bird
species, 103,648 research-grade observations).

Soft prior with a floor, never a filter. Two reasons: vagrants happen, and a
model that *cannot* emit a rare species will never let you discover one.
Observation density also measures where *people* record birds, not purely where
birds are, so treating it as ground truth would encode that sampling bias.

    weight = max(floor, (count / max_count) ** alpha)     alpha=0.35, floor=0.02

### D28. The prior is applied IN LOG SPACE, BEFORE temperature

This was a real bug, now fixed. `inference.py` originally multiplied the prior
into probabilities *after* softmax, which renormalises the distribution the
temperature was fitted against:

| method | ECE | note |
|---|---|---|
| temperature only | 0.017 | no prior |
| **temperature THEN prior** | **0.080** | the bug |
| prior in logits, T refit | 0.021 | the fix |
| vector scaling | 0.040 | worse than one temperature |

A prior over classes is additive in log space, which is what a logit is. Adding
`log(prior)` to the logits and fitting the temperature on the adjusted logits
keeps calibration essentially intact.

Vector scaling (per-class scale and bias) was also tried and is **worse** than a
single temperature — it overfits the validation split. One scalar wins.

### D29. A site prior cannot be evaluated on a geographically uniform test set

Evaluated naively, the prior looked harmful: accuracy 0.685 → 0.622. That is an
artefact. The test set holds 234 *Cinnyris afer* images; a Rondebosch feeder
would see roughly one.

Re-weighting test images by local observation density — which simulates the
deployment mix — reverses it: **0.693 → 0.724 (+3.1pp)**.

The lesson generalises: any deployment-specific prior must be measured against a
deployment-weighted sample, or you are measuring the wrong distribution.

### D30. Compute trimming: what worked and what did not

Three candidates measured. Only one survived.

**Input resolution — REJECTED.** Fine-grained species ID depends on exactly the
detail downsampling discards:

| input | MACs | rel. cost | accuracy |
|---|---|---|---|
| 224px | 0.720G | 1.00 | 0.684 |
| 192px | 0.527G | 0.73 | 0.645 |
| 160px | 0.366G | 0.51 | 0.600 |
| 128px | 0.235G | 0.33 | 0.525 |

224→192 costs 3.9 points for 27% of the MACs. A poor trade.

**Pruning dead classes — REJECTED, not worth it.** 40 of 62 classes have zero
training data and leak 6.3% of probability mass on average (up to 46.6% on
individual images). But masking them leaves accuracy unchanged at 0.679 and
saves 51K parameters against a 5.9M backbone — under 1%.

Worth noting for later: the genus/family/guild fallback classes should arguably
not be softmax outputs at all. They are *computed* by summing species
probabilities during rollup, so having them as separate logits lets the model
put mass directly on `cinnyris_indet`, competing with the species it is meant to
aggregate. Revisit at fine-tune time.

**INT8 quantisation — WORKS, but is not free.** See A22.
