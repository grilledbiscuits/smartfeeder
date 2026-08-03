"""Evaluate the open-set failsafe against real out-of-distribution photographs.

Fits every novelty scorer on in-distribution TRAINING features, calibrates the
threshold on in-distribution VALIDATION features, and then measures how much
genuinely out-of-distribution material it rejects -- squirrels, bees, agamas,
cats -- none of which any scorer has ever seen.

Reports, per scorer:

* **AUROC** -- threshold-free separability.
* **TPR at 1/5/10% FPR** -- the operational number. "At a 5% cost in missed real
  bird visits, what share of intruders is rejected?"
* **Per-group recall** -- broken out by mammal / insect / reptile, because a
  detector that catches baboons and misses carpenter bees is not much use at a
  nectar feeder, where the bees are the ones actually competing for the food.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from birdcam.config import Config, load_config
from birdcam.models.novelty import (
    EnergyScorer,
    KNNScorer,
    MahalanobisScorer,
    MaxSoftmaxScorer,
    auroc,
)

logger = logging.getLogger(__name__)


def _train_linear_head(cfg, Xtr, ytr, epochs: int = 200):
    """Small linear head over frozen features; returns a logits function."""
    import torch
    import torch.nn as nn

    h = nn.Linear(Xtr.shape[1], len(cfg.taxon_classes))
    opt = torch.optim.AdamW(h.parameters(), lr=0.01, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        ce(h(xt), yt).backward()
        opt.step()
    h.eval()

    def logits(X):
        with torch.no_grad():
            return h(torch.tensor(X, dtype=torch.float32)).numpy()

    return logits


def evaluate(cfg: Config, feature_file: str, fprs=(0.01, 0.05, 0.10)) -> dict:
    """Fit, calibrate and score every detector. Returns a JSON-able summary."""
    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest

    emb_dir = cfg.path("embeddings_dir") / "sweep"
    X = np.load(emb_dir / feature_file)

    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
        ood_rows = list(m.iter_rows("tier='OOD' AND status='downloaded'"))

    if len(X) != len(items):
        raise RuntimeError(
            f"feature/label misalignment: {len(X)} features vs {len(items)} in-distribution items. "
            "Re-extract features after changing the corpus."
        )

    split = np.array([i.split for i in items])
    y = np.array([i.taxon_index for i in items])
    tr, va, te = split == "train", split == "val", split == "test"

    ood_feats = _ood_features(cfg, ood_rows, feature_file)
    if ood_feats is None or len(ood_feats) == 0:
        raise RuntimeError("no OOD features; run fetch_ood + preprocess + extract_ood first")

    logits_fn = _train_linear_head(cfg, X[tr], y[tr])
    lg_va, lg_te, lg_ood = logits_fn(X[va]), logits_fn(X[te]), logits_fn(ood_feats)

    groups = _ood_groups(cfg, ood_rows)

    scorers = [
        MaxSoftmaxScorer(),
        EnergyScorer(),
        MahalanobisScorer(n_components=128),
        KNNScorer(k=10),
    ]

    results = []
    for sc in scorers:
        sc.fit(X[tr], y[tr])
        s_te = sc.score(X[te], lg_te)
        s_ood = sc.score(ood_feats, lg_ood)

        row = {
            "scorer": sc.name,
            "auroc": round(auroc(s_te, s_ood), 4),
            "tpr_at_fpr": {},
            "thresholds": {},
            "per_group": {},
        }
        for f in fprs:
            # Threshold from VAL, applied to test/OOD -- never fitted on test.
            thr = sc.calibrate(X[va], lg_va, target_fpr=f)
            row["tpr_at_fpr"][f"{f:.2f}"] = round(float((s_ood > thr).mean()), 4)
            row["thresholds"][f"{f:.2f}"] = round(thr, 4)
            # Realised in-distribution false-alarm rate on TEST, which is the
            # honest version of the target rate set on val.
            row.setdefault("realised_fpr_test", {})[f"{f:.2f}"] = round(
                float((s_te > thr).mean()), 4
            )

        thr5 = sc.calibrate(X[va], lg_va, target_fpr=0.05)
        for g in sorted(set(groups)):
            sel = np.array([x == g for x in groups])
            row["per_group"][g] = {
                "n": int(sel.sum()),
                "caught_at_5pct_fpr": round(float((s_ood[sel] > thr5).mean()), 4),
            }
        results.append(row)
        logger.info("%s AUROC %.4f", sc.name, row["auroc"])

    summary = {
        "feature_file": feature_file,
        "n_id_test": int(te.sum()),
        "n_ood": int(len(ood_feats)),
        "scorers": results,
    }
    out = cfg.path("reports_dir") / "open_set.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _ood_features(cfg: Config, ood_rows, feature_file: str) -> np.ndarray | None:
    """Load cached OOD features, extracting them with the same backbone if absent.

    Must use the identical backbone and transform as the in-distribution
    features, or the comparison is meaningless -- the detector would be
    measuring a difference in preprocessing rather than a difference in content.
    """
    path = cfg.path("embeddings_dir") / "sweep" / f"ood_{feature_file}"
    if path.is_file():
        feats = np.load(path)
        if len(feats) == len(ood_rows):
            return feats
        logger.warning(
            "cached OOD features have %d rows but the manifest has %d; re-extracting",
            len(feats),
            len(ood_rows),
        )

    import timm
    import torch
    from PIL import Image

    backbone_name = feature_file.rsplit("_", 1)[0]
    logger.info("extracting OOD features with %s (%d images)", backbone_name, len(ood_rows))
    model = timm.create_model(backbone_name, pretrained=True, num_classes=0).eval()
    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    paths = [
        cfg.path("processed_dir")
        / r["scientific_name"].lower().replace(" ", "_")
        / f"{r['image_id'].replace(':', '_')}.jpg"
        for r in ood_rows
    ]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} OOD images not preprocessed (e.g. {missing[0]}). "
            "Run birdcam.data.preprocess first."
        )

    with torch.inference_mode():
        probe = model(torch.zeros(1, 3, 224, 224))
    feats = np.zeros((len(paths), probe.shape[1]), dtype=np.float32)
    batch, idx = [], []
    with torch.inference_mode():
        for i, p_ in enumerate(paths):
            with Image.open(p_) as im:
                batch.append(transform(im.convert("RGB")))
            idx.append(i)
            if len(batch) == 32 or i == len(paths) - 1:
                feats[idx] = model(torch.stack(batch)).float().numpy()
                batch, idx = [], []
    np.save(path, feats)
    return feats


def _ood_groups(cfg: Config, ood_rows) -> list[str]:
    import yaml

    with (cfg.root / "config" / "ood.yaml").open(encoding="utf-8") as fh:
        ood_cfg = yaml.safe_load(fh)
    by_name = {t["scientific_name"]: t.get("group", "other") for t in ood_cfg["taxa"]}
    return [by_name.get(r["scientific_name"], "other") for r in ood_rows]


def print_report(summary: dict) -> None:
    print(
        f"\nOpen-set failsafe: {summary['n_id_test']} in-distribution test images "
        f"vs {summary['n_ood']} real OOD photographs\n"
        "(no scorer has ever seen an OOD example -- all fitted on target birds only)\n"
    )
    hdr = f"{'scorer':<16}{'AUROC':>8}{'TPR@1%':>9}{'TPR@5%':>9}{'TPR@10%':>9}   cost"
    print(hdr)
    print("-" * len(hdr))
    cost = {
        "max_softmax": "free (logits)",
        "energy": "free (logits)",
        "mahalanobis": "1 matmul, ~66KB",
        "knn": "5k refs resident",
    }
    for r in sorted(summary["scorers"], key=lambda x: -x["auroc"]):
        t = r["tpr_at_fpr"]
        print(
            f"{r['scorer']:<16}{r['auroc']:>8.3f}{t['0.01']:>9.3f}{t['0.05']:>9.3f}"
            f"{t['0.10']:>9.3f}   {cost.get(r['scorer'], '')}"
        )

    best = max(summary["scorers"], key=lambda x: x["auroc"])
    print(f"\nper-group recall at 5% false-alarm rate ({best['scorer']}):")
    for g, v in sorted(best["per_group"].items()):
        print(f"  {g:<10} n={v['n']:<5} caught {v['caught_at_5pct_fpr']:.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate the open-set failsafe.")
    ap.add_argument("--features", default="tf_efficientnetv2_b0.in1k_9318.npy")
    args = ap.parse_args()
    print_report(evaluate(load_config(), args.features))


if __name__ == "__main__":
    main()
