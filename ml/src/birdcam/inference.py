"""The single decision function: raw model outputs -> one honest label.

This is what the capture application calls. It composes four layers, in order,
and the order matters:

1. **Open-set failsafe.** If the novelty score exceeds its calibrated threshold,
   emit ``unknown`` and stop. Nothing else runs. A squirrel must never reach the
   taxonomic logic, because that logic is incapable of expressing "not a bird" --
   it will always return the least-bad species.

2. **Range prior, then temperature scaling.** The prior is added to the logits
   in log space -- NOT multiplied into probabilities afterwards, which would
   renormalise the distribution the temperature was fitted against and wreck
   calibration (measured: ECE 0.017 -> 0.080 the wrong way round, 0.021 the
   right way). The raw head is roughly 2x overconfident, so uncalibrated
   probabilities would make every threshold below meaningless.

3. **Rollup.** Species -> genus -> family -> guild. If no species clears its
   threshold but the summed genus probability does, emit the genus. The model
   must be able to say "a sunbird, I don't know which" instead of guessing.

4. **Abstain.** If nothing clears even the guild threshold, emit ``uncertain``.

The distinction between ``unknown`` and ``uncertain`` is deliberate and worth
keeping:

* ``unknown``  -- "this is not a thing I know about" (a squirrel, a hand, rain)
* ``uncertain`` -- "this is probably a bird, but I cannot pin it down"

They call for different downstream behaviour. ``uncertain`` visits are the
valuable ones for the data flywheel: they are birds worth a human look.
``unknown`` triggers are mostly noise, though a sustained run of them is itself
informative -- it usually means something has changed at the feeder.

All of this runs OUTSIDE the exported ONNX graph. Thresholds, the prior and the
temperature are therefore retunable without touching the model artefact -- which
matters on either candidate board, and matters a great deal on a Pi 5 + Hailo
where changing the graph means a recompile on a separate x86 toolchain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from birdcam.config import Config

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
UNCERTAIN = "uncertain"


@dataclass
class Decision:
    """One classification outcome, with the reasoning kept attached."""

    label: str
    level: str  # species | genus | family | guild | negative | unknown | uncertain
    confidence: float
    sex_label: str | None = None
    sex_confidence: float = 0.0
    novelty_score: float = 0.0
    is_unknown: bool = False
    # Whether this label is on the capture allowlist. Set by the Classifier,
    # which is the only thing that knows the configured targets.
    is_capture_target: bool = False
    # Top species candidates regardless of the outcome; useful for the
    # active-learning export and for debugging a surprising decision.
    top_k: list[tuple[str, float]] = field(default_factory=list)

    @property
    def should_record(self) -> bool:
        """Whether this decision alone justifies committing video.

        This used to be `level in {"species", "genus"}`, which was wrong twice
        over. `decide()` labelled every winning class "species" regardless of
        what it was, so `empty_feeder` and `insect` -- classes that exist
        precisely to suppress recording -- returned should_record=True. And a
        level test cannot distinguish a target species from a Tier C bystander
        anyway: recording is a question about WHICH taxon, not how specific the
        answer is.

        Membership of the capture allowlist is now the only criterion.
        """
        return not self.is_unknown and self.is_capture_target


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Classifier:
    """Composes calibration, the open-set failsafe and the taxonomic rollup."""

    def __init__(
        self,
        cfg: Config,
        novelty_scorer=None,
        temperature: float = 1.0,
        range_prior: dict[str, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.novelty = novelty_scorer
        self.temperature = temperature
        self.taxon_classes = cfg.taxon_classes
        self.sex_classes = cfg.sex_classes
        self.rollup = cfg.taxonomy_cfg["rollup"]
        self.thresholds = self.rollup["thresholds"]
        self.per_class = self.rollup.get("per_class_thresholds") or {}
        # Range prior: a per-class weight, applied as an additive log offset on
        # the logits (see decide()). Soft, never a filter -- a species that does
        # not occur here is downweighted, not banned, so a vagrant remains
        # detectable.
        self.range_prior = range_prior or {}
        self._log_prior = None
        if self.range_prior:
            w = np.ones(len(cfg.taxon_classes), dtype=np.float64)
            for label, weight in self.range_prior.items():
                idx = cfg.taxon_class_index.get(label)
                if idx is not None:
                    # Floor the weight: log(0) is -inf and would make the class
                    # permanently unreachable, which is a filter, not a prior.
                    w[idx] = max(float(weight), 1e-6)
            self._log_prior = np.log(w)

        self._species_idx = {
            s.slug: cfg.taxon_class_index[s.slug]
            for s in cfg.species
            if s.slug in cfg.taxon_class_index
        }
        self._genus_of = {s.slug: s.genus for s in cfg.species}
        self._genus_to_family = cfg.genus_to_family
        self._family_to_guild = cfg.family_to_guild

        # What KIND of class each label is. The winning class must be described
        # by what it is, not by the branch that returned it: `empty_feeder`
        # winning the argmax is not a species-level identification, and
        # `cinnyris_indet` winning directly is a genus answer even though no
        # rollup happened.
        head = cfg.taxonomy_cfg["taxon_head"]
        self._kind: dict[str, str] = {s.slug: "species" for s in cfg.species}
        for slug in head["genus_fallback"]:
            self._kind[slug] = "genus"
        for slug in head["family_fallback"]:
            self._kind[slug] = "family"
        for slug in head["guild_fallback"]:
            self._kind[slug] = "guild"
        for slug in head["negative"]:
            self._kind[slug] = "negative"

        # Capture allowlist: the labels worth committing video for. Tier A is
        # the deployment target; Tier B/C exist as hard negatives so the model
        # can recognise a bystander without recording it. A genus fallback
        # qualifies only if that genus actually contains a target, so
        # `cinnyris_indet` ("one of the double-collareds") records while
        # `nectariniidae_indet` ("some sunbird") does not -- the latter is too
        # vague to be worth storage.
        targets = {s.slug for s in cfg.species_by_tier("A")}
        target_genera = {s.genus for s in cfg.species_by_tier("A")}
        self._capture_targets = set(targets)
        for slug, spec in head["genus_fallback"].items():
            if spec.get("genus") in target_genera:
                self._capture_targets.add(slug)

    # -- rollup helpers --------------------------------------------------------

    def _group_mass(self, probs: np.ndarray, member_slugs: list[str]) -> float:
        return float(
            sum(probs[self._species_idx[s]] for s in member_slugs if s in self._species_idx)
        )

    def _genus_members(self, genus: str) -> list[str]:
        return [slug for slug, g in self._genus_of.items() if g == genus]

    def _family_members(self, family: str) -> list[str]:
        return [
            slug for slug, g in self._genus_of.items() if self._genus_to_family.get(g) == family
        ]

    def _guild_members(self, guild: str) -> list[str]:
        return [
            slug
            for slug, g in self._genus_of.items()
            if self._family_to_guild.get(self._genus_to_family.get(g, ""), "") == guild
        ]

    # -- main ------------------------------------------------------------------

    def decide(
        self,
        taxon_logits: np.ndarray,
        sex_logits: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> Decision:
        """Turn one frame's raw outputs into a single labelled decision."""
        taxon_logits = np.asarray(taxon_logits).reshape(-1)

        # --- 1. open-set failsafe, BEFORE any taxonomic reasoning -------------
        novelty_score = 0.0
        if self.novelty is not None:
            f = features.reshape(1, -1) if features is not None else None
            novelty_score = float(self.novelty.score(f, taxon_logits.reshape(1, -1))[0])
            if self.novelty.threshold is not None and novelty_score > self.novelty.threshold:
                return Decision(
                    label=UNKNOWN,
                    level="unknown",
                    confidence=0.0,
                    novelty_score=novelty_score,
                    is_unknown=True,
                )

        # --- 2. range prior, IN LOG SPACE, BEFORE temperature -----------------
        # Order matters and getting it wrong is expensive. Multiplying the prior
        # into probabilities *after* softmax renormalises a distribution the
        # temperature was fitted against, and wrecks calibration: measured
        # 2026-08-04, ECE 0.017 -> 0.080. Adding log(prior) to the logits and
        # fitting the temperature on the adjusted logits keeps it at 0.021.
        #
        # This is also the principled form -- a prior over classes is additive
        # in log space, which is what a logit is.
        if self._log_prior is not None:
            taxon_logits = taxon_logits + self._log_prior

        # --- 3. calibrated probabilities -------------------------------------
        probs = softmax(taxon_logits, self.temperature)

        order = np.argsort(probs)[::-1][:5]
        top_k = [(self.taxon_classes[i], float(probs[i])) for i in order]

        sex_label, sex_conf = None, 0.0
        if sex_logits is not None:
            sp = softmax(np.asarray(sex_logits).reshape(-1), self.temperature)
            j = int(sp.argmax())
            sex_label, sex_conf = self.sex_classes[j], float(sp[j])

        def out(label, level, conf):
            return Decision(
                label=label,
                level=level,
                confidence=conf,
                sex_label=sex_label,
                sex_confidence=sex_conf,
                novelty_score=novelty_score,
                is_capture_target=label in self._capture_targets,
                top_k=top_k,
            )

        # --- 4. rollup: species -> genus -> family -> guild -------------------
        best = int(np.argmax(probs))
        best_label = self.taxon_classes[best]
        species_thr = self.per_class.get(best_label, self.thresholds["species"])
        if probs[best] >= species_thr:
            # The level reflects what the winning class IS. A negative class
            # clearing the threshold is a confident "nothing to record here",
            # not a species identification.
            return out(best_label, self._kind.get(best_label, "species"), float(probs[best]))

        if best_label in self._genus_of:
            genus = self._genus_of[best_label]
            mass = self._group_mass(probs, self._genus_members(genus))
            if mass >= self.thresholds["genus"]:
                slug = f"{genus.lower()}_indet"
                # Only emit a genus fallback that actually exists as a class;
                # single-species genera have none by design and roll to family.
                if slug in self.cfg.taxon_class_index:
                    return out(slug, "genus", mass)

            family = self._genus_to_family.get(genus)
            if family:
                mass = self._group_mass(probs, self._family_members(family))
                if mass >= self.thresholds["family"]:
                    return out(f"{family.lower()}_indet", "family", mass)

                guild = self._family_to_guild.get(family)
                if guild:
                    mass = self._group_mass(probs, self._guild_members(guild))
                    if mass >= self.thresholds["guild"]:
                        return out(f"{guild}_indet", "guild", mass)

        # --- 5. abstain -------------------------------------------------------
        return out(UNCERTAIN, "uncertain", float(probs[best]))

    def vote(self, decisions: list[Decision]) -> Decision:
        """Combine per-frame decisions across one visit into a single label.

        A visit yields many frames; voting across them beats any single frame,
        and per the brief improves real-world accuracy more than an architecture
        change would.

        `unknown` frames are counted but do NOT outvote confident bird frames --
        a squirrel passing behind a feeding sunbird should not suppress the
        sunbird. Only if unknown is the *majority* is the whole visit unknown.
        """
        if not decisions:
            return Decision(UNCERTAIN, "uncertain", 0.0)

        n_unknown = sum(d.is_unknown for d in decisions)
        if n_unknown > len(decisions) / 2:
            return Decision(
                UNKNOWN,
                "unknown",
                0.0,
                is_unknown=True,
                novelty_score=float(np.mean([d.novelty_score for d in decisions])),
            )

        named = [d for d in decisions if not d.is_unknown and d.level != "uncertain"]
        if not named:
            return Decision(UNCERTAIN, "uncertain", 0.0)

        # Confidence-weighted vote: a few high-confidence frames should outweigh
        # many marginal ones.
        scores: dict[str, float] = {}
        levels: dict[str, str] = {}
        for d in named:
            scores[d.label] = scores.get(d.label, 0.0) + d.confidence
            levels[d.label] = d.level
        label = max(scores, key=lambda k: scores[k])
        members = [d for d in named if d.label == label]
        return Decision(
            label=label,
            level=levels[label],
            confidence=float(np.mean([d.confidence for d in members])),
            sex_label=_majority([d.sex_label for d in members if d.sex_label]),
            sex_confidence=float(np.mean([d.sex_confidence for d in members])),
            novelty_score=float(np.mean([d.novelty_score for d in decisions])),
            top_k=members[0].top_k,
        )


def _majority(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])
