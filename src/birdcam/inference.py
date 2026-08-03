"""The single decision function: raw model outputs -> one honest label.

This is what the capture application calls. It composes four layers, in order,
and the order matters:

1. **Open-set failsafe.** If the novelty score exceeds its calibrated threshold,
   emit ``unknown`` and stop. Nothing else runs. A squirrel must never reach the
   taxonomic logic, because that logic is incapable of expressing "not a bird" --
   it will always return the least-bad species.

2. **Temperature scaling.** Calibrates confidence before any threshold sees it.
   The raw head is roughly 2x overconfident, so uncalibrated probabilities would
   make every threshold below meaningless.

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

All of this runs OUTSIDE the exported ONNX graph, so thresholds can be retuned
without a Hailo recompile.
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
    level: str  # species | genus | family | guild | unknown | uncertain
    confidence: float
    sex_label: str | None = None
    sex_confidence: float = 0.0
    novelty_score: float = 0.0
    is_unknown: bool = False
    # Top species candidates regardless of the outcome; useful for the
    # active-learning export and for debugging a surprising decision.
    top_k: list[tuple[str, float]] = field(default_factory=list)

    @property
    def should_record(self) -> bool:
        """Whether this decision alone justifies committing video."""
        return not self.is_unknown and self.level in {"species", "genus"}


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
        # Range prior: multiplicative per-species weight, so a species that does
        # not occur at this site is downweighted rather than banned outright.
        self.range_prior = range_prior or {}

        self._species_idx = {
            s.slug: cfg.taxon_class_index[s.slug]
            for s in cfg.species
            if s.slug in cfg.taxon_class_index
        }
        self._genus_of = {s.slug: s.genus for s in cfg.species}
        self._genus_to_family = cfg.genus_to_family
        self._family_to_guild = cfg.family_to_guild

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

        # --- 2. calibrated probabilities -------------------------------------
        probs = softmax(taxon_logits, self.temperature)

        # --- range prior ------------------------------------------------------
        if self.range_prior:
            adj = probs.copy()
            for slug, idx in self._species_idx.items():
                adj[idx] *= self.range_prior.get(slug, 1.0)
            total = adj.sum()
            if total > 0:
                probs = adj / total

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
                top_k=top_k,
            )

        # --- 3. rollup: species -> genus -> family -> guild -------------------
        best = int(np.argmax(probs))
        best_label = self.taxon_classes[best]
        species_thr = self.per_class.get(best_label, self.thresholds["species"])
        if probs[best] >= species_thr:
            return out(best_label, "species", float(probs[best]))

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

        # --- 4. abstain -------------------------------------------------------
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
