"""Tests for the open-set failsafe and the inference decision function.

The property that matters most: a squirrel must never be given a species name.
The taxonomic rollup cannot express "not a bird" -- it will always return the
least-bad species -- so the novelty check has to run first and short-circuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdcam.config import load_config
from birdcam.inference import UNCERTAIN, UNKNOWN, Classifier, Decision, softmax
from birdcam.models.novelty import (
    EnergyScorer,
    KNNScorer,
    MahalanobisScorer,
    MaxSoftmaxScorer,
    auroc,
    tpr_at_fpr,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def synthetic():
    """Two well-separated in-distribution clusters plus a distant OOD blob."""
    rng = np.random.RandomState(0)
    a = rng.randn(300, 32) + np.r_[np.ones(4) * 3, np.zeros(28)]
    b = rng.randn(300, 32) + np.r_[np.zeros(4), np.ones(4) * 3, np.zeros(24)]
    ood = rng.randn(200, 32) * 0.5 + 12.0
    X = np.vstack([a, b]).astype(np.float32)
    y = np.r_[np.zeros(300, dtype=int), np.ones(300, dtype=int)]
    return X, y, ood.astype(np.float32)


# --- scorers ------------------------------------------------------------------


@pytest.mark.parametrize("cls", [MahalanobisScorer, KNNScorer])
def test_feature_scorers_separate_ood(synthetic, cls):
    X, y, ood = synthetic
    s = cls().fit(X, y)
    assert auroc(s.score(X), s.score(ood)) > 0.95


def test_energy_scorer_separates_ood_from_logits(synthetic):
    """Energy should rise when no class fires strongly."""
    X, y, ood = synthetic
    confident = np.tile(np.r_[10.0, 0.0, 0.0], (len(X), 1))
    flat = np.zeros((len(ood), 3))
    s = EnergyScorer()
    assert s.score(X, confident).mean() < s.score(ood, flat).mean()


def test_scorers_are_fitted_only_on_in_distribution_data(synthetic):
    """The detector must never require OOD examples to work.

    Fitting on known intruders would make it a closed-set classifier for those
    intruders and useless against the first novel one.
    """
    X, y, ood = synthetic
    s = KNNScorer().fit(X, y)  # ood never passed
    assert auroc(s.score(X), s.score(ood)) > 0.9


def test_calibrate_hits_the_requested_false_alarm_rate(synthetic):
    X, y, ood = synthetic
    s = KNNScorer().fit(X, y)
    for target in (0.01, 0.05, 0.20):
        s.calibrate(X, target_fpr=target)
        realised = float((s.score(X) > s.threshold).mean())
        assert abs(realised - target) < 0.02, (target, realised)


def test_uncalibrated_scorer_refuses_to_decide(synthetic):
    X, y, _ = synthetic
    s = KNNScorer().fit(X, y)
    with pytest.raises(RuntimeError, match="not calibrated"):
        s.is_unknown(X)


def test_max_softmax_is_blind_to_confident_ood(synthetic):
    """Documents WHY a naive confidence threshold is not an adequate failsafe.

    The real failure mode is not that OOD inputs look uncertain -- it is that
    they frequently look *confident*. A network that has only ever seen birds
    has no mechanism for reporting low confidence on a squirrel; it simply picks
    the nearest bird and commits.

    Here half the OOD batch produces confident predictions. Max-softmax scores
    those as perfectly in-distribution and catches none of them, while a
    feature-space detector still flags them on distance alone.

    Measured on real data: max-softmax catches 17.8% of intruders at a 5%
    false-alarm rate, against kNN's 90.9%.
    """
    X, y, ood = synthetic
    id_logits = np.tile(np.r_[8.0, 0.0], (len(X), 1))
    ood_logits = np.tile(np.r_[8.0, 0.0], (len(ood), 1))
    ood_logits[: len(ood) // 2] = np.r_[1.0, 0.9]  # half look uncertain

    weak = MaxSoftmaxScorer()
    weak_tpr, _ = tpr_at_fpr(weak.score(X, id_logits), weak.score(ood, ood_logits), 0.05)
    # The confidently-wrong half is invisible to it.
    assert weak_tpr <= 0.55

    strong = KNNScorer().fit(X, y)
    strong_tpr, _ = tpr_at_fpr(strong.score(X), strong.score(ood), 0.05)
    assert strong_tpr > weak_tpr, "feature-space detector must beat max-softmax"


def test_auroc_matches_hand_computed_value():
    """Guard the rank-sum implementation, including ties."""
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.5


def test_pca_is_optional_and_preserves_ranking(synthetic):
    X, y, ood = synthetic
    full = MahalanobisScorer(n_components=None).fit(X, y)
    red = MahalanobisScorer(n_components=8).fit(X, y)
    assert auroc(full.score(X), full.score(ood)) > 0.9
    assert auroc(red.score(X), red.score(ood)) > 0.9


# --- decision function --------------------------------------------------------


class _AlwaysNovel:
    name = "stub"
    threshold = 0.0

    def score(self, features, logits=None):
        return np.ones(len(features))


class _NeverNovel:
    name = "stub"
    threshold = 10.0

    def score(self, features, logits=None):
        return np.zeros(len(features))


def test_novelty_short_circuits_before_any_species_is_named(cfg):
    """THE critical property: a squirrel must never get a species label."""
    clf = Classifier(cfg, novelty_scorer=_AlwaysNovel())
    logits = np.zeros(len(cfg.taxon_classes))
    logits[cfg.taxon_class_index["cinnyris_chalybeus"]] = 50.0  # screamingly confident
    d = clf.decide(logits, features=np.zeros(8))
    assert d.label == UNKNOWN
    assert d.is_unknown
    assert not d.should_record


def test_confident_species_passes_when_not_novel(cfg):
    clf = Classifier(cfg, novelty_scorer=_NeverNovel())
    logits = np.zeros(len(cfg.taxon_classes))
    logits[cfg.taxon_class_index["cinnyris_chalybeus"]] = 50.0
    d = clf.decide(logits, features=np.zeros(8))
    assert d.label == "cinnyris_chalybeus"
    assert d.level == "species"
    assert d.should_record


def test_flat_logits_abstain_rather_than_guess(cfg):
    clf = Classifier(cfg, novelty_scorer=_NeverNovel())
    d = clf.decide(np.zeros(len(cfg.taxon_classes)), features=np.zeros(8))
    assert d.label == UNCERTAIN
    assert not d.should_record


def test_unknown_and_uncertain_are_distinct(cfg):
    """Different meanings, different downstream handling."""
    novel = Classifier(cfg, novelty_scorer=_AlwaysNovel()).decide(
        np.zeros(len(cfg.taxon_classes)), features=np.zeros(8)
    )
    flat = Classifier(cfg, novelty_scorer=_NeverNovel()).decide(
        np.zeros(len(cfg.taxon_classes)), features=np.zeros(8)
    )
    assert novel.label == UNKNOWN and novel.is_unknown
    assert flat.label == UNCERTAIN and not flat.is_unknown


def test_genus_rollup_when_species_is_split(cfg):
    """Two confusable congeners splitting the mass should emit the genus."""
    clf = Classifier(cfg, novelty_scorer=_NeverNovel())
    logits = np.full(len(cfg.taxon_classes), -20.0)
    logits[cfg.taxon_class_index["cinnyris_chalybeus"]] = 3.0
    logits[cfg.taxon_class_index["cinnyris_afer"]] = 2.9
    d = clf.decide(logits, features=np.zeros(8))
    assert d.level in {"genus", "family", "guild"}
    if d.level == "genus":
        assert d.label == "cinnyris_indet"


def test_range_prior_downweights_implausible_species(cfg):
    """A species that does not occur at this site should lose to one that does."""
    logits = np.full(len(cfg.taxon_classes), -20.0)
    logits[cfg.taxon_class_index["cinnyris_neergaardi"]] = 5.0
    logits[cfg.taxon_class_index["cinnyris_chalybeus"]] = 4.0
    plain = Classifier(cfg, novelty_scorer=_NeverNovel()).decide(logits, features=np.zeros(8))
    primed = Classifier(
        cfg,
        novelty_scorer=_NeverNovel(),
        range_prior={"cinnyris_neergaardi": 0.001, "cinnyris_chalybeus": 1.0},
    ).decide(logits, features=np.zeros(8))
    assert plain.top_k[0][0] == "cinnyris_neergaardi"
    assert primed.top_k[0][0] == "cinnyris_chalybeus"


def test_temperature_changes_confidence_not_ranking(cfg):
    logits = np.array([3.0, 2.0, 1.0])
    hot, cold = softmax(logits, 1.0), softmax(logits, 3.0)
    assert hot.argmax() == cold.argmax()
    assert hot.max() > cold.max()


# --- track voting -------------------------------------------------------------


def test_vote_emits_one_label_per_visit(cfg):
    clf = Classifier(cfg, novelty_scorer=_NeverNovel())
    frames = [
        Decision("cinnyris_chalybeus", "species", 0.9),
        Decision("cinnyris_chalybeus", "species", 0.8),
        Decision("cinnyris_afer", "species", 0.4),
    ]
    assert clf.vote(frames).label == "cinnyris_chalybeus"


def test_a_passing_squirrel_does_not_suppress_a_feeding_bird(cfg):
    """Minority unknown frames must not veto a confident bird track."""
    clf = Classifier(cfg, novelty_scorer=_NeverNovel())
    frames = [
        Decision("cinnyris_chalybeus", "species", 0.9),
        Decision("cinnyris_chalybeus", "species", 0.85),
        Decision(UNKNOWN, "unknown", 0.0, is_unknown=True),
    ]
    assert clf.vote(frames).label == "cinnyris_chalybeus"


def test_majority_unknown_makes_the_whole_visit_unknown(cfg):
    clf = Classifier(cfg, novelty_scorer=_NeverNovel())
    frames = [
        Decision(UNKNOWN, "unknown", 0.0, is_unknown=True),
        Decision(UNKNOWN, "unknown", 0.0, is_unknown=True),
        Decision("cinnyris_chalybeus", "species", 0.9),
    ]
    v = clf.vote(frames)
    assert v.label == UNKNOWN and v.is_unknown


def test_vote_on_empty_track_abstains(cfg):
    assert Classifier(cfg, novelty_scorer=_NeverNovel()).vote([]).label == UNCERTAIN
