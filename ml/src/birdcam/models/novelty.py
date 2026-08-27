"""Open-set failsafe: flag anything the classifier has no business naming.

## The problem

A softmax over 62 classes always returns one of the 62. Point the camera at a
grey squirrel and the model reports a bird -- often confidently, because nothing
in training ever taught it that "none of the above" is an option. For a system
whose job is deciding what to record, that is the difference between a useful
log and a folder of squirrels labelled *Cinnyris chalybeus*.

## The approach

Detectors here are fitted on IN-DISTRIBUTION TRAINING FEATURES ONLY. None of
them ever sees an out-of-distribution example. That is the whole point: a
detector trained on squirrels learns "squirrel", and the next intruder is a
mongoose, a hand, a blown leaf or rain on the lens. Fitting only on what the
target birds look like means anything sufficiently unlike them is flagged,
including things nobody enumerated.

Three scorers, deliberately spanning a cost range:

* **Energy** -- ``-logsumexp(logits)``. Free: the logits already exist. Better
  than max-softmax because it keeps the magnitude information that softmax
  normalises away.
* **Mahalanobis** -- distance to the nearest class-conditional Gaussian under a
  shared covariance. Costs one small matmul. Optional PCA keeps the precision
  matrix tiny for the Pi.
* **kNN** -- distance to the k-th nearest training feature. Strongest in the
  literature, but needs the training features resident. Included as a reference
  ceiling; use it to judge what the cheap scorers give up.

## The threshold is an operational choice, not a statistical one

It is set on the in-distribution VALIDATION split to a target false-alarm rate:
"flag at most X% of genuine birds as unknown". That is the knob worth exposing,
because it states the trade in the operator's terms -- how many real visits you
will miss in order to stop recording squirrels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@dataclass
class NoveltyScorer:
    """Base: higher score == more novel == more likely NOT a target bird."""

    name: str = "base"
    threshold: float | None = None
    target_fpr: float = 0.05

    def fit(self, features: np.ndarray, labels: np.ndarray) -> NoveltyScorer:
        raise NotImplementedError

    def score(self, features: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def calibrate(
        self,
        features: np.ndarray,
        logits: np.ndarray | None = None,
        target_fpr: float | None = None,
    ) -> float:
        """Set the threshold from IN-DISTRIBUTION validation data.

        Calibrating on val rather than test keeps the test estimate honest, and
        calibrating on in-distribution data only means no OOD examples are
        needed to deploy -- which matters, because the real intruders at a
        specific feeder are not knowable in advance.
        """
        fpr = target_fpr if target_fpr is not None else self.target_fpr
        s = self.score(features, logits)
        # Flag the top `fpr` fraction of genuine birds; everything above this
        # score is called unknown.
        self.threshold = float(np.quantile(s, 1.0 - fpr))
        self.target_fpr = fpr
        return self.threshold

    def is_unknown(self, features: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError(f"{self.name} not calibrated; call calibrate() first")
        return self.score(features, logits) > self.threshold

    # -- serialisation ---------------------------------------------------------
    #
    # The on-disk format is defined HERE and nowhere else. It previously lived
    # in two places -- the exporter wrote a bespoke .npz and the capture service
    # read it back by assigning to the private `_ref` and re-running the private
    # `_normalise`. That worked only because normalising an already-normalised
    # vector is a no-op; any change to what `fit()` stores would have broken the
    # deployment path silently, at the one point in the system whose whole job
    # is to fail safe.

    def _state(self) -> dict[str, np.ndarray]:
        """Subclass hook: arrays this scorer needs to score again."""
        return {}

    def _load_state(self, data) -> None:
        """Subclass hook: restore what `_state` emitted."""

    def save(self, path, **meta) -> Path:
        """Write scorer, threshold and provenance as one file.

        The threshold travels WITH the reference vectors deliberately. Splitting
        them -- vectors in a bundle, threshold in a service config -- is an
        invitation for the two to drift, and a novelty gate carrying someone
        else's threshold fails in the direction of silently passing intruders.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.threshold is None:
            raise RuntimeError(f"refusing to save an uncalibrated {self.name} scorer")
        # Hyperparameters travel too. Restoring only the threshold and letting
        # the constructor supply defaults for k / max_reference / temperature
        # silently rescores everything: a scorer fitted at k=5 reloaded at the
        # default k=10 flipped 116 of 200 decisions in test.
        params = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_") and f.name not in ("name", "threshold")
        }
        payload = {
            "scorer": np.array(self.name),
            "threshold": np.array(float(self.threshold)),
            "params": np.array(json.dumps(params)),
            "meta": np.array(json.dumps(meta)),
            **self._state(),
        }
        np.savez_compressed(path, **payload)
        logger.info("wrote %s scorer to %s (threshold %.4f)", self.name, path, self.threshold)
        return path


def load_scorer(path) -> NoveltyScorer:
    """Reconstruct a saved scorer, dispatching on the stored name.

    Counterpart to `NoveltyScorer.save`. Callers get a working scorer with its
    threshold already set; nothing outside this module needs to know the file
    layout or touch a private attribute.
    """
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    name = str(z["scorer"])
    kinds = {c.name: c for c in (EnergyScorer, MaxSoftmaxScorer, MahalanobisScorer, KNNScorer)}
    if name not in kinds:
        raise ValueError(f"{path} holds unknown scorer {name!r}; expected one of {sorted(kinds)}")
    params = json.loads(str(z["params"])) if "params" in z.files else {}
    known = {f.name for f in fields(kinds[name])}
    unknown = set(params) - known
    if unknown:
        raise ValueError(
            f"{path} carries parameters this build does not know: {sorted(unknown)}. "
            "The scorer definition changed since it was saved; re-export it."
        )
    scorer = kinds[name](threshold=float(z["threshold"]), **params)
    scorer._load_state(z)
    return scorer


def scorer_meta(path) -> dict:
    """Provenance recorded alongside a saved scorer, without loading arrays."""
    return json.loads(str(np.load(Path(path), allow_pickle=False)["meta"]))


@dataclass
class EnergyScorer(NoveltyScorer):
    """Free-at-inference scorer over the logits we already compute.

        E(x) = -T * logsumexp(logits / T)

    Low energy == the model found strong evidence for some class. High energy ==
    nothing fired, which is what an out-of-distribution input looks like.
    Preferred over max-softmax because softmax normalises away the overall
    magnitude, which is exactly the signal that matters here.
    """

    name: str = "energy"
    temperature: float = 1.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> EnergyScorer:
        return self  # nothing to fit

    def score(self, features: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if logits is None:
            raise ValueError("EnergyScorer needs logits")
        z = _as_float64(logits) / self.temperature
        m = z.max(axis=1, keepdims=True)
        lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))).squeeze(1)
        return -self.temperature * lse


@dataclass
class MaxSoftmaxScorer(NoveltyScorer):
    """Baseline: 1 - max softmax probability.

    Included because it is the obvious thing to reach for, and because having it
    in the comparison shows whether the cleverer scorers earn their cost.
    """

    name: str = "max_softmax"

    def fit(self, features: np.ndarray, labels: np.ndarray) -> MaxSoftmaxScorer:
        return self

    def score(self, features: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if logits is None:
            raise ValueError("MaxSoftmaxScorer needs logits")
        z = _as_float64(logits)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return 1.0 - p.max(axis=1)


@dataclass
class MahalanobisScorer(NoveltyScorer):
    """Distance to the nearest class-conditional Gaussian, shared covariance.

    Cheap to deploy: class means plus one precision matrix. With PCA the
    precision matrix is `n_components` squared -- a few tens of KB at 128
    components, against ~6.5MB at a raw 1280 dims.
    """

    name: str = "mahalanobis"
    n_components: int | None = 128
    shrinkage: float = 0.01

    _mean: np.ndarray | None = field(default=None, repr=False)
    _components: np.ndarray | None = field(default=None, repr=False)
    _class_means: np.ndarray | None = field(default=None, repr=False)
    _precision: np.ndarray | None = field(default=None, repr=False)

    def _state(self) -> dict[str, np.ndarray]:
        if self._class_means is None or self._precision is None:
            raise RuntimeError("refusing to save an unfitted mahalanobis scorer")
        out = {
            "mean": self._mean,
            "class_means": self._class_means,
            "precision": self._precision,
        }
        # `_components` is None when no PCA was applied; npz cannot store None,
        # so its absence on load means exactly that.
        if self._components is not None:
            out["components"] = self._components
        return out

    def _load_state(self, data) -> None:
        self._mean = _as_float64(data["mean"])
        self._class_means = _as_float64(data["class_means"])
        self._precision = _as_float64(data["precision"])
        self._components = _as_float64(data["components"]) if "components" in data.files else None

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = _as_float64(x) - self._mean
        return x @ self._components if self._components is not None else x

    def fit(self, features: np.ndarray, labels: np.ndarray) -> MahalanobisScorer:
        X = _as_float64(features)
        self._mean = X.mean(axis=0, keepdims=True)
        Xc = X - self._mean

        if self.n_components and self.n_components < X.shape[1]:
            # PCA via SVD. Keeps the precision matrix small enough to ship.
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            self._components = vt[: self.n_components].T
            Xc = Xc @ self._components
        else:
            self._components = None

        classes = np.unique(labels)
        means = []
        centred = np.empty_like(Xc)
        for c in classes:
            sel = labels == c
            mu = Xc[sel].mean(axis=0)
            means.append(mu)
            centred[sel] = Xc[sel] - mu
        self._class_means = np.stack(means)

        # Shared covariance across classes, shrunk toward a scaled identity so
        # the inverse stays well-conditioned when a class has few examples --
        # which is the norm here, not the exception.
        cov = np.cov(centred, rowvar=False)
        cov += self.shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self._precision = np.linalg.inv(cov)
        logger.info(
            "mahalanobis fitted: %d classes, %d dims (from %d)",
            len(classes),
            Xc.shape[1],
            features.shape[1],
        )
        return self

    def score(self, features: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if self._class_means is None:
            raise RuntimeError("not fitted")
        Z = self._project(features)
        # Squared Mahalanobis distance to every class mean; keep the minimum.
        out = np.empty((len(Z), len(self._class_means)))
        for i, mu in enumerate(self._class_means):
            d = Z - mu
            out[:, i] = np.einsum("ij,jk,ik->i", d, self._precision, d)
        return out.min(axis=1)


@dataclass
class KNNScorer(NoveltyScorer):
    """Distance to the k-th nearest training feature, on L2-normalised vectors.

    Reference ceiling. Needs the training features (or a subsample) resident at
    inference, so it is the most expensive option -- included to show what the
    cheap scorers give up.
    """

    name: str = "knn"
    k: int = 10
    # Measured 2026-08-02 (18,146 in-distribution vs 2,486 real OOD photos):
    #   refs=5000 -> AUROC 0.979, TPR@5%FPR 0.909, 25.6MB, 12.8 MFLOPs/frame
    #   refs=1000 -> AUROC 0.981, TPR@5%FPR 0.916,  5.1MB,  2.6 MFLOPs/frame
    # Cutting references costs nothing, so take the cheap one. Against the
    # backbone's 1440 MFLOPs that is 0.18% overhead -- the failsafe is free.
    #
    # DO NOT reduce dimensionality to save memory. PCA wrecks it:
    #   1280 dims -> 0.916 | 256 -> 0.573 | 128 -> 0.475 | 64 -> 0.343
    # The novelty signal lives in the low-variance directions that PCA discards,
    # which is exactly what you would expect -- the high-variance directions
    # encode what separates the bird species from each other, not what separates
    # birds from squirrels.
    max_reference: int = 1000

    _ref: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        x = _as_float64(x)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> KNNScorer:
        X = self._normalise(features)
        if len(X) > self.max_reference:
            rng = np.random.RandomState(0)
            X = X[rng.choice(len(X), self.max_reference, replace=False)]
        self._ref = X
        return self

    def _state(self) -> dict[str, np.ndarray]:
        if self._ref is None:
            raise RuntimeError("refusing to save an unfitted knn scorer")
        # Already L2-normalised by `fit`. Saved that way so loading is a plain
        # read with no preprocessing to get wrong.
        return {"reference": self._ref.astype(np.float32)}

    def _load_state(self, data) -> None:
        self._ref = _as_float64(data["reference"])

    def score(self, features: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if self._ref is None:
            raise RuntimeError("not fitted")
        Q = self._normalise(features)
        out = np.empty(len(Q))
        # Chunked: the full distance matrix would not fit comfortably in the
        # dev machine's spare RAM.
        for i in range(0, len(Q), 512):
            chunk = Q[i : i + 512]
            sim = chunk @ self._ref.T
            kth = np.partition(sim, -self.k, axis=1)[:, -self.k]
            out[i : i + len(chunk)] = 1.0 - kth
        return out


# --- evaluation ---------------------------------------------------------------


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for separating OOD (positive) from in-distribution (negative).

    Computed via the rank-sum identity, which needs no sklearn and handles ties.
    """
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within tie groups.
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tpr_at_fpr(id_scores: np.ndarray, ood_scores: np.ndarray, fpr: float) -> tuple[float, float]:
    """Fraction of OOD caught when flagging `fpr` of genuine birds.

    This is the operationally meaningful number: at a 5% cost in missed real
    visits, what share of squirrels is rejected?
    """
    thr = float(np.quantile(id_scores, 1.0 - fpr))
    return float((ood_scores > thr).mean()), thr


def main() -> None:
    raise NotImplementedError("novelty.py is a library module; see eval/open_set.py")
