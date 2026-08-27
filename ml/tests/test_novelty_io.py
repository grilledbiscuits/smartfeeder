"""Round-trip tests for novelty scorer serialisation.

The open-set gate is the one component whose entire job is to fail safe, so a
scorer that reloads as a *slightly different* scorer is the worst possible
defect: it keeps working, and quietly applies the wrong operating point.

This file exists because the first implementation did exactly that.
`load_scorer` restored the threshold but let the constructor supply default
hyperparameters, so a scorer fitted at k=5 came back at the default k=10 and
flipped 116 of 200 decisions. Nothing raised. Every test here asserts on
behaviour after a round trip, never on the file's contents.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdcam.models.novelty import (
    EnergyScorer,
    KNNScorer,
    MahalanobisScorer,
    MaxSoftmaxScorer,
    load_scorer,
    scorer_meta,
)

RS = np.random.RandomState(0)
FEATS = RS.randn(300, 32)
LABELS = RS.randint(0, 3, 300)
QUERY = np.random.RandomState(1).randn(200, 32)
LOGITS = np.random.RandomState(2).randn(200, 6)


def _fitted(kind):
    """A fitted, calibrated scorer with NON-DEFAULT hyperparameters.

    Non-default on purpose: defaults would hide exactly the bug this guards.
    """
    if kind == "knn":
        s = KNNScorer(k=5, max_reference=100).fit(FEATS, LABELS)
        s.threshold = 0.5
        return s, (QUERY, None)
    if kind == "mahalanobis":
        s = MahalanobisScorer(n_components=8).fit(FEATS, LABELS)
        s.threshold = 0.5
        return s, (QUERY, None)
    if kind == "energy":
        s = EnergyScorer(temperature=2.0, threshold=-1.0)
        return s, (None, LOGITS)
    s = MaxSoftmaxScorer(threshold=-1.0)
    return s, (None, LOGITS)


ALL = ["knn", "mahalanobis", "energy", "max_softmax"]


@pytest.mark.parametrize("kind", ALL)
def test_roundtrip_scores_identically(tmp_path, kind) -> None:
    """The whole contract: a reloaded scorer must be the same function."""
    s, args = _fitted(kind)
    r = load_scorer(s.save(tmp_path / "s.npz"))
    assert np.allclose(s.score(*args), r.score(*args))


@pytest.mark.parametrize("kind", ALL)
def test_roundtrip_preserves_type_and_threshold(tmp_path, kind) -> None:
    s, _ = _fitted(kind)
    r = load_scorer(s.save(tmp_path / "s.npz"))
    assert type(r) is type(s)
    assert r.threshold == s.threshold


def test_roundtrip_preserves_k(tmp_path) -> None:
    """The exact regression: k=5 must not come back as the default k=10."""
    s, _ = _fitted("knn")
    assert s.k == 5
    r = load_scorer(s.save(tmp_path / "s.npz"))
    assert r.k == 5, "hyperparameters were not restored; scores will differ silently"


def test_roundtrip_preserves_no_decision_flips(tmp_path) -> None:
    """Score closeness is not enough -- what matters is the unknown/known call."""
    s, args = _fitted("knn")
    r = load_scorer(s.save(tmp_path / "s.npz"))
    assert not np.any(s.is_unknown(*args) != r.is_unknown(*args))


def test_mahalanobis_without_pca_roundtrips(tmp_path) -> None:
    """`_components` is None when PCA is off, and npz cannot store None."""
    s = MahalanobisScorer(n_components=None).fit(FEATS, LABELS)
    s.threshold = 0.5
    r = load_scorer(s.save(tmp_path / "s.npz"))
    assert r.n_components is None
    assert np.allclose(s.score(QUERY), r.score(QUERY))


def test_refuses_to_save_uncalibrated(tmp_path) -> None:
    """An unthresholded gate cannot decide anything; saving one hides that."""
    s = KNNScorer(k=5).fit(FEATS, LABELS)
    with pytest.raises(RuntimeError, match="uncalibrated"):
        s.save(tmp_path / "s.npz")


def test_refuses_to_save_unfitted(tmp_path) -> None:
    s = KNNScorer(threshold=0.5)
    with pytest.raises(RuntimeError, match="unfitted"):
        s.save(tmp_path / "s.npz")


def test_meta_is_readable_without_loading_arrays(tmp_path) -> None:
    s, _ = _fitted("knn")
    p = s.save(tmp_path / "s.npz", backbone="test", note="provenance")
    assert scorer_meta(p) == {"backbone": "test", "note": "provenance"}


def test_unknown_scorer_name_is_rejected(tmp_path) -> None:
    p = tmp_path / "bogus.npz"
    np.savez_compressed(
        p, scorer=np.array("nonsense"), threshold=np.array(0.5),
        params=np.array("{}"), meta=np.array("{}"),
    )  # fmt: skip
    with pytest.raises(ValueError, match="unknown scorer"):
        load_scorer(p)


def test_unrecognised_parameter_is_rejected(tmp_path) -> None:
    """A bundle from a newer scorer definition must fail loudly, not partially."""
    p = tmp_path / "future.npz"
    np.savez_compressed(
        p, scorer=np.array("energy"), threshold=np.array(-1.0),
        params=np.array('{"temperature": 2.0, "warp_factor": 9}'), meta=np.array("{}"),
    )  # fmt: skip
    with pytest.raises(ValueError, match="does not know"):
        load_scorer(p)
