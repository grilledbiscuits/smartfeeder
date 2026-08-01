"""Fast loop: train heads on cached features. Target <10s per run.

This is where most of the actual work happens. Everything is a matrix operation
on features already in memory, so a full train+evaluate cycle finishes in
seconds and questions can be answered by experiment rather than by argument.

Two heads:

* **taxon** -- ordinary cross-entropy.
* **sex/plumage** -- MASKED cross-entropy over admissible classes. An annotated
  male has no evidence for breeding vs eclipse, so its loss is
  ``-log(p_male_breeding + p_male_eclipse)``. Samples with an exact label reduce
  to ordinary cross-entropy. This trains "is male" without inventing a plumage
  state that no source records.

Every accuracy figure is reported with a Wilson 95% confidence interval. On the
classes that matter most -- female Cinnyris -- the test sets are small enough
that a point estimate alone would be actively misleading.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from birdcam.config import Config, load_config
from birdcam.train.cache_embeddings import load_cached

logger = logging.getLogger(__name__)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because it stays sane at small n
    and near 0 or 1 -- exactly the regime the female classes sit in.
    """
    if total == 0:
        return (0.0, 1.0)
    p = correct / total
    d = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / d
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / d
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class ClassResult:
    label: str
    support: int
    correct: int
    precision: float
    recall: float
    f1: float
    ci_low: float
    ci_high: float
    verdict: str = ""


@dataclass
class RunResult:
    taxon_accuracy: float
    taxon_ci: tuple[float, float]
    sex_accuracy: float
    sex_ci: tuple[float, float]
    per_class: list[ClassResult] = field(default_factory=list)
    confusion: np.ndarray | None = None
    labels: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _standardise(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True) + 1e-6
    return ((train - mu) / sd, *[(o - mu) / sd for o in others])


def train_heads(cfg: Config, stem: str, epochs: int | None = None, head_type: str | None = None):
    """Train both heads on cached features and evaluate on the test split."""
    import torch
    import torch.nn as nn

    from birdcam.utils.runtime import setup_torch

    setup_torch(cfg)
    fl = cfg.train_cfg["fast_loop"]
    epochs = epochs or fl["epochs"]
    head_type = head_type or fl["head_type"]

    feats, items = load_cached(cfg, stem)
    splits = np.array([i["split"] for i in items])
    taxon_y = np.array([i["taxon_index"] for i in items], dtype=np.int64)
    sex_mask = np.array([i["sex_mask"] for i in items], dtype=np.float32)

    tr, va, te = (splits == "train"), (splits == "val"), (splits == "test")
    if te.sum() == 0:
        raise RuntimeError("test split is empty; re-run preprocess to assign splits")

    Xtr, Xva, Xte = _standardise(feats[tr], feats[va], feats[te])
    t0 = time.monotonic()

    dev = torch.device("cpu")
    Xtr_t = torch.tensor(Xtr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev)
    ytr_t = torch.tensor(taxon_y[tr], device=dev)
    mtr_t = torch.tensor(sex_mask[tr], device=dev)

    n_taxon = len(cfg.taxon_classes)
    n_sex = len(cfg.sex_classes)
    dim = feats.shape[1]

    def make_head(out_dim: int) -> nn.Module:
        if head_type == "mlp":
            h = fl["mlp_hidden"]
            return nn.Sequential(
                nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, out_dim)
            )
        return nn.Linear(dim, out_dim)

    taxon_head = make_head(n_taxon)
    sex_head = make_head(n_sex)
    params = list(taxon_head.parameters()) + list(sex_head.parameters())
    opt = torch.optim.AdamW(params, lr=fl["lr"], weight_decay=fl["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    sex_weight = cfg.train_cfg["train"]["loss"]["sex_weight"]

    for _ep in range(epochs):
        taxon_head.train()
        sex_head.train()
        opt.zero_grad()
        loss = ce(taxon_head(Xtr_t), ytr_t)

        # Masked partial-label loss: -log(sum of admissible probabilities).
        logp = torch.log_softmax(sex_head(Xtr_t), dim=1)
        # logsumexp over admissible classes only; -inf elsewhere.
        masked = logp.masked_fill(mtr_t == 0, float("-inf"))
        loss = loss + sex_weight * (-torch.logsumexp(masked, dim=1)).mean()

        loss.backward()
        opt.step()
        sched.step()

    taxon_head.eval()
    sex_head.eval()
    with torch.inference_mode():
        taxon_pred = taxon_head(Xte_t).argmax(1).numpy()
        sex_logits = sex_head(Xte_t)
        sex_pred = sex_logits.argmax(1).numpy()

    elapsed = time.monotonic() - t0

    y_true = taxon_y[te]
    correct = int((taxon_pred == y_true).sum())
    total = int(te.sum())
    acc = correct / total

    # A sex prediction counts as correct if it lands anywhere in the admissible
    # set -- for a masked male, either plumage state is right, because the
    # ground truth genuinely does not distinguish them.
    te_mask = sex_mask[te]
    sex_correct = int(te_mask[np.arange(len(sex_pred)), sex_pred].sum())
    sex_acc = sex_correct / total

    present = sorted(set(y_true) | set(taxon_pred))
    labels = [cfg.taxon_classes[i] for i in present]
    conf = np.zeros((len(present), len(present)), dtype=int)
    pos = {c: i for i, c in enumerate(present)}
    for t_, p_ in zip(y_true, taxon_pred, strict=True):
        conf[pos[t_], pos[p_]] += 1

    min_n = cfg.taxonomy_cfg["class_size_policy"]["min_test_images_for_verdict"]
    per_class: list[ClassResult] = []
    for c in present:
        tp = int(((taxon_pred == c) & (y_true == c)).sum())
        fp = int(((taxon_pred == c) & (y_true != c)).sum())
        fn = int(((taxon_pred != c) & (y_true == c)).sum())
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lo, hi = wilson_interval(tp, support)
        verdict = "" if support >= min_n else f"INSUFFICIENT DATA (n={support} < {min_n})"
        per_class.append(
            ClassResult(cfg.taxon_classes[c], support, tp, prec, rec, f1, lo, hi, verdict)
        )

    return RunResult(
        taxon_accuracy=acc,
        taxon_ci=wilson_interval(correct, total),
        sex_accuracy=sex_acc,
        sex_ci=wilson_interval(sex_correct, total),
        per_class=per_class,
        confusion=conf,
        labels=labels,
        seconds=elapsed,
    )


def print_report(cfg: Config, res: RunResult) -> None:
    budget = cfg.train_cfg["fast_loop"]["target_seconds_per_run"]
    flag = "" if res.seconds <= budget else f"  (OVER {budget}s BUDGET)"
    print(f"\ntrain+eval: {res.seconds:.2f}s{flag}")
    print(
        f"taxon accuracy : {res.taxon_accuracy:.3f}  "
        f"[95% CI {res.taxon_ci[0]:.3f}-{res.taxon_ci[1]:.3f}]"
    )
    print(
        f"sex accuracy   : {res.sex_accuracy:.3f}  "
        f"[95% CI {res.sex_ci[0]:.3f}-{res.sex_ci[1]:.3f}]   "
        "(masked: any admissible class counts as correct)"
    )

    print(f"\n{'class':<28}{'n':>5}{'prec':>7}{'rec':>7}{'f1':>7}   95% CI on recall")
    print("-" * 78)
    for c in sorted(res.per_class, key=lambda x: -x.support):
        note = f"   {c.verdict}" if c.verdict else ""
        print(
            f"{c.label:<28}{c.support:>5}{c.precision:>7.3f}{c.recall:>7.3f}"
            f"{c.f1:>7.3f}   [{c.ci_low:.2f}-{c.ci_high:.2f}]{note}"
        )

    if res.confusion is not None and len(res.labels) <= 12:
        print("\nconfusion matrix (rows=true, cols=pred):")
        w = max(len(x) for x in res.labels)
        print(" " * (w + 2) + "".join(f"{x[:7]:>8}" for x in res.labels))
        for i, lab in enumerate(res.labels):
            print(f"{lab:<{w}}  " + "".join(f"{v:>8}" for v in res.confusion[i]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="Fast loop: train heads on cached features.")
    ap.add_argument("--stem", required=True, help="embedding stem, e.g. local_convnext_tiny...")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--head", default=None, choices=["linear", "mlp"])
    args = ap.parse_args()

    cfg = load_config()
    res = train_heads(cfg, args.stem, epochs=args.epochs, head_type=args.head)
    print_report(cfg, res)


if __name__ == "__main__":
    main()
