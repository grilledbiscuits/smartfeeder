# Assumptions

Things I have assumed, inferred, or could not verify — collected here rather
than buried in comments. **Each needs your confirmation.** Ordered by how much
damage a wrong assumption would do.

Legend: 🔴 blocks or invalidates results · 🟡 shapes design · 🟢 minor

---

## 🔴 A1. The sex head is data-starved, and it is exactly where you said it would hurt

Measured against the live iNaturalist API on 2026-07-31 (research grade, South
Africa, with photos):

| Species | Observations | Male | Female | Sex-annotated |
|---|---|---|---|---|
| *Cinnyris chalybeus* | 6203 | 530 | **161** | 11.2% |
| *Anthobaphes violacea* | 3617 | 885 | 207 | 30.2% |
| *Nectarinia famosa* | 4405 | 606 | 202 | 18.4% |
| *Promerops cafer* | 4701 | 186 | **59** | 5.3% |
| *Chalcomitra amethystina* | 3526 | 591 | 295 | 25.3% |
| *Cinnyris afer* | 3021 | 512 | **108** | 20.5% |

Only 5–30% of observations carry a Sex annotation, and males outnumber females
roughly 3:1 — photographer bias toward bright breeding males.

**Consequence:** after dedup and a grouped split, female *C. chalybeus* and
female *C. afer* yield roughly 16–24 test images each. That is not enough to
settle whether they are separable. The Phase 5 fast loop will report
"insufficient data to decide" for these classes rather than a verdict (D11).

**This is the project's binding constraint — not the architecture.** The
photographs exist; the annotations do not. If you want the merge question
actually answered this round, manually sexing a few hundred existing images is
by far the highest-value use of your time. Phase 5 will quantify exactly how
many per class.

**Please confirm:** should the pipeline include a labelling-assist export
(uncertainty-ranked crops, ready to sex) as part of Phase 5 rather than waiting
for Phase 8?

## 🔴 A2. No source annotates eclipse plumage

The iNaturalist Sex controlled term has exactly three values (Male, Female,
Cannot Be Determined) and Life Stage offers Adult and Juvenile. GBIF's `sex`
field is binary. Wikimedia and Flickr have no structured annotation at all.

I have therefore assumed `male_breeding` vs `male_eclipse` **cannot be sourced
automatically at all**, and implemented the masked partial-label loss (D3)
instead of fabricating labels.

**Please confirm:** is that acceptable, or do you have another source in mind?
The Macaulay Library request (Phase 2) is the realistic route to real plumage
annotations — their media *is* annotated for age and sex — which is why the
draft email matters more than it might appear.

## 🔴 A3. There is a git repository over your entire home directory

`/home/grilledbiscuits/.git` exists with zero commits, and tracks nothing yet.
It looks like an accidental `git init`.

I did not touch it, and gave `smartfeeder/` its own repository instead (D1).

**This is a standing hazard:** a stray `git add -A` run from your home directory
would stage `.ssh/` private keys, `.gnupg/`, `.claude.json`, browser profiles and
Thunderbird mail. **Recommend you delete it** (`rm -rf ~/.git`) once you have
confirmed it holds nothing you want — but that is your call to make, not mine.

## 🟡 A4. GBIF places the Fiscal Flycatcher in *Sigelus*, not *Melaenornis*

`GET /species/match?name=Melaenornis silens` returns `Sigelus silens`, genus
*Sigelus*, matchType EXACT (via synonymy). Family is Muscicapidae either way, so
rollup is unaffected.

I kept **Melaenornis silens** in `species.yaml` — current IOC usage and the name
you supplied — and will record GBIF's own genus separately during resolution.
`taxa.py` will treat a derived-genus/backbone-genus mismatch as a warning
requiring an explicit acknowledgement, not a silent pass.

**Please confirm** you are happy retaining the IOC name.

## 🟡 A5. Rollup thresholds are placeholders

`species: 0.55, genus: 0.65, family: 0.75, guild: 0.80` in `taxonomy.yaml` are
**guesses**, chosen only to be monotonically increasing with generality. They
are not derived from anything.

Phase 7 populates `per_class_thresholds` from real precision-recall curves. Do
not read anything into the current values.

## 🟡 A6. Class-size policy numbers are judgement calls

`min_species_images: 100` (fold into genus), `warn_species_images: 250`,
`min_test_images_for_verdict: 50`. Reasoned but not empirical — the 100 figure
matches the threshold you asked to be flagged in the Phase 2 report.

Under this policy, *A. reichenowi* (33 global) and *C. neergaardi* (32 global)
will certainly fold into `anthreptes_indet`… except *Anthreptes* is a
single-species genus here, so they fold to `nectariniidae_indet` instead. Worth
knowing before you see it happen.

## 🟡 A7. Monomorphic species list is from general ornithological knowledge

`monomorphic_forced_na` in `taxonomy.yaml` lists 13 Tier C species plus both
*Cyanomitra* as visually monomorphic, forcing their sex label to
`not_applicable` so the head is not trained on noise.

This is **my knowledge, not a cited source**. *Cyanomitra veroxii* and
*C. olivacea* in particular are worth a second look — they show subtle
dimorphism that a field guide would list even if a feeder camera could not
resolve it. You are far better placed to check this list than I am.

## 🟢 A8. Local fast-loop backbone choice

`convnext_tiny.in12k_ft_in1k` was chosen as a CPU-feasible compromise. I have
**not** benchmarked its extraction throughput on this machine — the 30–45 min
estimate for 20k images is from your brief, not measured. Phase 5 will measure
it and report the real figure.

## 🟢 A9. `max_images_per_species: 1500`

Caps the head of the distribution so *C. chalybeus* does not swamp training.
Arbitrary. Interacts with class-balanced sampling and focal loss, both of which
Phase 6 reports on, so it can be revisited with evidence.

## 🟢 A10. Augmentation parameters are unvalidated

Motion blur kernel 3–11px, occlusion up to 30% area, JPEG quality 30–75. Chosen
to mimic the deployment failure modes you described. Until real feeder captures
exist there is nothing to validate them against — the domain gap remains the
project's largest technical risk, and these numbers are a hypothesis about it.

---

## 🔴 A11. First real scores: females are worse, but not in the way expected

Vertical slice, 2026-08-01. Tier A only (6 species), 3,573 images after dedup,
linear probe on frozen `convnext_tiny.in12k_ft_in1k` features. Taxon accuracy
**0.684** [95% CI 0.644-0.722]. Test recall broken out by sex:

| Species | Sex | n | Recall | 95% CI |
|---|---|---|---|---|
| *C. chalybeus* | male | 57 | 0.544 | 0.42-0.67 |
| *C. chalybeus* | female | 39 | 0.359 | 0.23-0.52 |
| *C. afer* | male | 63 | 0.667 | 0.54-0.77 |
| *C. afer* | female | 28 | 0.250 | 0.13-0.43 |

The male/female gap is real — the CIs for male and female *C. afer* do not
overlap. But **where the errors go differs by sex**, and that was not expected:

- **Male** errors concentrate on the sibling species (male *chalybeus* → *afer*
  14 of 26 errors; male *afer* → *chalybeus* 14 of 21).
- **Female** errors scatter across all species (female *chalybeus* → *afer* 11,
  *famosa* 7, *amethystina* 5; female *afer* → *chalybeus* 6, *violacea* 5,
  *famosa* 4).

**This matters for the merge decision.** Merging *C. chalybeus* and *C. afer*
into `cinnyris_double_collared_indet` would recover most *male* errors but only
about a third of *female* errors. The female problem is not primarily
sibling confusion — it looks like weak features from drab plumage. **Merging
would not fix it.**

Both female classes fall below `min_test_images_for_verdict: 50`, so the
pipeline correctly refuses to issue a merge verdict. Treat the above as a
direction, not a conclusion.

**Caveats, all of which inflate or deflate this number:**
- Linear probe on ImageNet features, not a fine-tune. A real fine-tune should
  improve absolute accuracy substantially.
- Six classes only. No Tier C, so the false-positive question is untested.
- The corpus over-samples annotated observations (D12), so it is harder than the
  wild distribution — the 0.684 is pessimistic as a deployment estimate and
  meaningless as a class-prior estimate.
- Still web photographs. The feeder-camera domain gap is entirely unmeasured.

---

## 🔴 A12. CORRECTION to A11 — merging the double-collared pair IS now justified

A11 concluded, on 3,573 images, that merging *C. chalybeus* and *C. afer* "would
not fix" the female problem because female errors scattered rather than
concentrating on the sibling. **With 9,318 images that conclusion is wrong and
is withdrawn.**

Measured on `tf_efficientnetv2_b0`, test split:

| | before merge | after merge | gain |
|---|---|---|---|
| overall | 0.727 | 0.786 | +5.9pp |
| **females only** (n=276) | 0.551 | 0.630 | **+8.0pp** |
| males only (n=758) | 0.785 | 0.846 | +6.1pp |

45% of all errors on the double-collared pair are mutual confusion between the
two. Merging now helps females **more** than males, the reverse of A11.

What changed is sample size, not the model. The earlier female error counts
(11 and 6) were too small to distinguish "scattered" from "concentrated"; at
n=48–58 per class the concentration is clear. This is precisely the failure mode
`min_test_images_for_verdict` exists to prevent, and I drew a conclusion from
under-powered data anyway. Treat A11's directional claim as retracted.

**Recommendation:** add `cinnyris_double_collared_indet` as a merged class and
let the rollup emit it when neither species clears its threshold. This is the
brief's "correct outcome, not a failure" case.

## 🔴 A13. *Cinnyris afer* is the single worst Tier A class

Female recall 0.208 (`tf_efficientnetv2_b0`) and 0.167 (`convnext_tiny`), against
0.552/0.621 for female *C. chalybeus*. The errors are overwhelmingly one-way:
female *afer* → *chalybeus* (18 and 23 respectively), not the reverse.

The classifier is collapsing *afer* into *chalybeus*. That asymmetry suggests a
prior/imbalance effect on top of genuine visual similarity, so class-balanced
sampling and focal loss (both already configurable, Phase 6) should be evaluated
against it specifically before concluding the pair is inseparable.

## 🟡 A14. EfficientNetV2 Hailo compatibility is UNVERIFIED

`tf_efficientnetv2_b0` uses fused-MBConv blocks and squeeze-excite. Both are
standard convolutional constructs and should compile, but **I have not run the
Hailo Dataflow Compiler** — it is an x86 toolchain requiring a Developer Zone
account, and is not installed here.

If it turns out SE blocks compile poorly on the Hailo-8L, `efficientnet_b0`
(0.39G MACs, 0.691 taxon) and `mobilenetv3_large_100` (0.21G, 0.690) are the
fallbacks, at roughly -4pp taxon and -4 to -8pp female. Worth compiling all
three early rather than discovering this at deployment.

## 🟡 A15. More data helped substantially — and is probably still helping

Tier A taxon accuracy on `convnext_tiny` went **0.684 → 0.788** when the corpus
grew 3,573 → 9,318 images (+10.4pp for 2.6x data). The curve has not visibly
flattened.

The cap is now `max_images_per_species: 1500`, and several species hit it. Raising
it is the cheapest remaining accuracy lever — cheaper than any architecture
change measured so far — though it will not help the *sex* head, which is
limited by annotation availability, not photograph availability.

---

## 🔴 A16. Adding Tier C cost Tier A accuracy — as expected, and it was worth it

With Tier A alone (6 classes): Tier A macro recall **0.735**, female **0.577**.
With Tier C added (24 classes): Tier A macro recall **0.686**, female **0.471**.

That is −4.9pp on Tier A and −10.6pp on females. The drop is real and is the
price of the model being *able* to say "not a sunbird". The earlier Tier-A-only
number was flattering because the model was never asked to reject anything.

What Tier C buys, measured on the test split:

* Tier C bird called a nectarivore: **9.4%** [7.9–11.1]
* Tier A bird called a non-target: **10.4%** [8.9–12.1]

Worst offender by a distance: **Speckled Mousebird (*Colius striatus*), 40%
called a nectarivore** (n=60). Plausible — long tail and slender body read like a
Cape Sugarbird. Karoo Prinia (18.5%) and Cape Bulbul (17.6%) follow.

**Recommendation:** *Colius striatus* needs more data before deployment; at 40%
it is a live false-positive source at a Cape Town feeder, where mousebirds are
common.

## 🔴 A17. The OOD evaluation set is missing its most common members

`config/ood.yaml` covers animals. Real feeder footage will be dominated by
things iNaturalist cannot supply, because nobody uploads them as wildlife
observations:

* **empty feeder** — by far the most common motion trigger
* **rain, blown leaves, moving shadows, lens flare**
* **human hands and faces** (refilling the feeder; *Homo sapiens* has 0
  research-grade ZA records)

So the 90.9% catch rate is measured against the *animal* subset of the problem
only. It is necessary evidence, not sufficient. **The first week of real capture
data should be used to re-measure this**, and those frames are exactly what the
`unknown` log will collect automatically.

## 🟡 A18. Weakest OOD classes are the ones that matter most at a feeder

Per-taxon catch rate at 5% false alarm:

| taxon | caught |
|---|---|
| Agama atra | 0.982 |
| Trachylepis capensis | 0.974 |
| Rattus rattus / Papio ursinus / Felis catus | 0.955–0.958 |
| Papilio demodocus | 0.952 |
| **Sciurus carolinensis** (grey squirrel) | **0.930** |
| Apis mellifera | 0.916 |
| **Rhabdomys pumilio** (grass mouse) | **0.816** |
| **Xylocopa caffra** (carpenter bee) | **0.794** |

The two weakest are both small animals photographed at flowers — the same
context, scale and background as a feeding sunbird. Carpenter bees in particular
*share the feeder*, so they will be over-represented in real triggers relative to
this evaluation set. Expect the real-world figure to be worse than 90.9%.

## 🟡 A19. The 5% false-alarm rate is a placeholder

`target_false_alarm_rate: 0.05` means 5% of genuine bird frames are discarded as
unknown. End to end that yields: 69.1% of real bird frames trigger capture, 4.0%
of intruder frames do.

This is an operator preference, not a derived value. Track-level voting also
softens it — a visit yields many frames and only needs some to pass. Revisit once
you know whether you would rather miss visits or store squirrels.

---

## 🔴 A20. False triggers are close to solved. Confidence on real birds is not.

This inverts the priority. Measured 2026-08-04 at the chosen operating point
(novelty FAR 15%, per-class thresholds fitted for 80% precision):

* **OOD visits that trigger a capture: 0.4%.** At 20% FAR it is 0.0%.
* **Real bird visits that trigger a capture: 22.9%.**

The system is not recording squirrels. It is failing to record birds — roughly
three quarters of genuine visits never clear a threshold.

The cause is that frozen-feature confidence is poor on the hardest classes. At a
90% precision target the double-collared sunbirds only fire on 3–9% of their
frames; relaxing to 80% gets them to 20–37%:

| Tier A species (80% target) | threshold | test precision | recall |
|---|---|---|---|
| Amethyst Sunbird | 0.41 | 0.881 | 0.601 |
| Nectarinia famosa | 0.69 | 0.875 | 0.477 |
| Cinnyris afer | 0.56 | 0.841 | 0.406 |
| Anthobaphes violacea | 0.51 | 0.831 | 0.729 |
| Promerops cafer | 0.32 | 0.759 | 0.811 |
| **Cinnyris chalybeus** | **0.82** | **0.742** | **0.223** |

***Cinnyris chalybeus* is the worst case and also the most likely visitor** —
the commonest sunbird at a Cape Town feeder is the one the model is least
confident about. It cannot reach 80% precision at any threshold, and at 0.82 it
fires on under a quarter of its frames.

**The fix is not more thresholding.** These numbers come from a linear probe on
frozen ImageNet features; the ceiling is set by the features, not the head. The
highest-value next step is a real fine-tune (Phase 6, `train_full.py`, still
unwritten), which is now feasible locally — you have said you can train on the
laptop. Expect the biggest gain there, not from further tuning of what exists.

## 🟡 A21. "Highly likely visitor" is not yet encoded anywhere

The range prior exists in `inference.py` and is **empty**. Nothing currently
tells the model that a Cape Town feeder sees six species routinely and
*Cinnyris neergaardi* essentially never.

Populating it would raise precision on the real visitors for free — no
retraining, one config block. It needs your input, though: the weights should
reflect what your feeder actually sees, and I would be guessing. A first pass
could come from iNaturalist observation density within ~25km of the site.

---

## 🔴 A22. INT8 quantisation costs 5.1 accuracy points — not free

Measured 2026-08-04 on 800 test images, against a genuinely trained model
(backbone + frozen-feature heads, standardisation folded into the linear layer).
MinMax calibration, per-channel weights, 500 calibration images from train.

* **24.4 MB → 7.5 MB** (3.3x smaller)
* **accuracy 0.641 → 0.590 (−5.1pp)**
* the two models agree on only 72.2% of test images

Five of six Tier A classes degraded past the 2% flag threshold:

| class | FP32 | INT8 | delta |
|---|---|---|---|
| *Chalcomitra amethystina* | 0.714 | 0.607 | −10.7pp |
| *Anthobaphes violacea* | 0.803 | 0.697 | −10.6pp |
| *Promerops cafer* | 0.820 | 0.746 | −7.4pp |
| *Cinnyris afer* | 0.509 | 0.456 | −5.3pp |
| *Cinnyris chalybeus* | 0.554 | 0.523 | −3.2pp |

This is exactly the failure the brief predicted: nearly-identical classes sit
close together in feature space and INT8 smears them, **unevenly**. An averaged
delta would have hidden that *amethystina* and *violacea* took the worst of it.

**Not yet exhausted.** Only MinMax calibration was measured. Percentile and
Entropy calibration are usually kinder to outlier activations and often recover
much of the loss; that comparison was started and interrupted, and is the first
thing to finish. Mixed precision — keeping the final layers in FP32 — is the
next lever after that.

**Do not ship INT8 on these numbers.** If the board turns out to be a Pi 5 with
the AI HAT+, quantisation is mandatory anyway and this becomes a hard constraint
to engineer around. If it is a Pi 4B, it is a speed/accuracy dial you control.

## 🟡 A23. The first export with real weights was only just now possible

Every earlier ONNX export carried **randomly-initialised heads** — the graph was
valid, the cost figures were real, but the predictions were noise. Quantising it
measured noise against noise, which I caught only when the reported accuracy came
back at 0.05.

`export_from_frozen_head` now builds a genuinely trained model from the fast-loop
heads, folding the feature standardisation into the linear layer. Anything that
evaluates the exported artefact must use that path until `train_full.py` exists.
