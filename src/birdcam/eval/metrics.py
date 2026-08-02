"""Metrics: per-class scores, calibration, and sex-broken-out confusion.

Two things here are not standard boilerplate, and both exist because of how the
outputs get used downstream.

**Calibration is a first-class metric.** The capture application gates video
recording on confidence. A model that is 95% accurate but reports 0.99 on
everything is worse for that purpose than one that is 85% accurate and honest,
because the gate cannot discriminate. Expected calibration error is therefore
reported alongside accuracy, not as an afterthought.

**Confusion is broken out by sex.** A matrix aggregated over sexes hides exactly
the failure mode this project cares about -- abundant, easily-identified males
swamp the scarce females and the aggregate looks fine while the female classes
fail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Stays sane at small n and near 0 or 1."""
    if total == 0:
        return (0.0, 1.0)
    p = correct / total
    d = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / d
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / d
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class PerClass:
    label: str
    support: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    recall_ci: tuple[float, float]

    @property
    def verdict_reliable(self) -> bool:
        """Whether the support is large enough to state a conclusion.

        A 95% CI spanning tens of points is not a finding, and a merge or drop
        recommendation drawn from it would be guesswork wearing a number.
        """
        return self.support >= 50


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[PerClass]:
    out: list[PerClass] = []
    for i, name in enumerate(labels):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append(
            PerClass(name, support, tp, fp, fn, prec, rec, f1, wilson_interval(tp, support))
        )
    return out


def macro_recall(rows: list[PerClass], only: set[str] | None = None) -> float:
    """Macro (unweighted) recall.

    Macro rather than micro so an abundant class cannot mask a failing one --
    which is precisely what would happen here, where males outnumber females
    roughly three to one.
    """
    sel = [r for r in rows if r.support > 0 and (only is None or r.label in only)]
    return float(np.mean([r.recall for r in sel])) if sel else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        m[t, p] += 1
    return m


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """ECE plus the per-bin data needed to draw a reliability diagram.

    Confidence is the max predicted probability; a bin's gap is |accuracy -
    mean confidence|. ECE is the support-weighted mean gap.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (conf > lo) & (conf <= hi)
        n = int(sel.sum())
        if n == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "acc": None, "conf": None})
            continue
        acc = float(correct[sel].mean())
        mc = float(conf[sel].mean())
        ece += (n / len(conf)) * abs(acc - mc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": n, "acc": acc, "conf": mc})
    return float(ece), bins


@dataclass
class SexBreakdown:
    """Per-species accuracy split by sex label."""

    species: str
    by_sex: dict[str, tuple[int, int]] = field(default_factory=dict)  # sex -> (correct, n)

    def recall(self, sex: str) -> float | None:
        if sex not in self.by_sex:
            return None
        c, n = self.by_sex[sex]
        return c / n if n else None

    def ci(self, sex: str) -> tuple[float, float]:
        c, n = self.by_sex.get(sex, (0, 0))
        return wilson_interval(c, n)


def sex_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species: np.ndarray,
    sex_labels: np.ndarray,
) -> list[SexBreakdown]:
    """Accuracy per (species, sex). The aggregate matrix hides this."""
    out: list[SexBreakdown] = []
    for sp in sorted(set(species.tolist())):
        row = SexBreakdown(sp)
        for sx in sorted(set(sex_labels.tolist())):
            sel = (species == sp) & (sex_labels == sx)
            n = int(sel.sum())
            if n == 0:
                continue
            row.by_sex[sx] = (int((y_pred[sel] == y_true[sel]).sum()), n)
        out.append(row)
    return out


def error_flow(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], class_index: int, top: int = 3
) -> list[tuple[str, int]]:
    """Where a class's errors actually go.

    Distinguishes "confused with its sibling" from "scattered everywhere",
    which implies completely different remedies: merging sibling classes fixes
    the first and does nothing for the second.
    """
    sel = (y_true == class_index) & (y_pred != class_index)
    if sel.sum() == 0:
        return []
    vals, counts = np.unique(y_pred[sel], return_counts=True)
    pairs = sorted(zip(vals, counts, strict=True), key=lambda t: -t[1])[:top]
    return [(labels[v], int(c)) for v, c in pairs]


def main() -> None:
    raise NotImplementedError("metrics.py is a library module; nothing to run directly.")
