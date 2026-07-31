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
