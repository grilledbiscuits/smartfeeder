"""Fit operating thresholds: confidence quality, and false triggers from unknowns.

Two questions, both operational rather than academic:

1. **When the model names a likely visitor, how often is it right?**
   Not accuracy -- *precision at the confidence the system will actually act on*.
   A per-class threshold is fitted on the validation split so each Tier A
   species reaches a target precision, and the recall paid for it is reported.
   The thresholds are written into config/taxonomy.yaml.

2. **How often does an unknown thing trigger a capture?**
   Measured at the VISIT level, not the frame level. A visit is many frames and
   the track vote decides; a single unlucky frame does not commit video. Frame
   rates systematically overstate the problem.

Visits are simulated from real observation IDs -- genuine photo bursts of one
individual by one photographer -- rather than by grouping random frames. That is
the closest available stand-in for a feeder visit.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from birdcam.config import Config, load_config
from birdcam.eval.metrics import wilson_interval

logger = logging.getLogger(__name__)


@dataclass
class ClassThreshold:
    label: str
    common_name: str
    threshold: float
    precision: float
    recall: float
    support: int
    achievable: bool


def fit_class_thresholds(
    probs_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Config,
    target_precision: float = 0.90,
) -> list[ClassThreshold]:
    """Lowest threshold reaching `target_precision` on validation, per Tier A class.

    Lowest rather than highest: among thresholds meeting the precision target we
    want the one that keeps the most recall. Fitted on val so the test estimate
    stays honest.
    """
    out: list[ClassThreshold] = []
    for sp in cfg.species_by_tier("A"):
        idx = cfg.taxon_class_index.get(sp.slug)
        if idx is None:
            continue
        scores = probs_va[:, idx]
        pos = y_va == idx
        best: ClassThreshold | None = None
        for t in np.linspace(0.05, 0.99, 95):
            sel = scores >= t
            if sel.sum() < 5:
                continue
            prec = float(pos[sel].mean())
            rec = float((pos & sel).sum() / max(pos.sum(), 1))
            if prec >= target_precision:
                best = ClassThreshold(
                    sp.slug, sp.common_name, float(t), prec, rec, int(pos.sum()), True
                )
                break
        if best is None:
            # Target unreachable at any threshold: report the best precision
            # available rather than pretending a threshold exists.
            prec_at = []
            for t in np.linspace(0.05, 0.99, 95):
                sel = scores >= t
                if sel.sum() >= 5:
                    prec_at.append((float(pos[sel].mean()), float(t)))
            if prec_at:
                p_, t_ = max(prec_at)
                sel = scores >= t_
                rec = float((pos & sel).sum() / max(pos.sum(), 1))
                best = ClassThreshold(sp.slug, sp.common_name, t_, p_, rec, int(pos.sum()), False)
            else:
                best = ClassThreshold(
                    sp.slug, sp.common_name, 0.99, 0.0, 0.0, int(pos.sum()), False
                )
        out.append(best)
    return out


def evaluate_thresholds(
    probs_te: np.ndarray, y_te: np.ndarray, cfg: Config, thresholds: list[ClassThreshold]
) -> list[dict]:
    """Apply val-fitted thresholds to test. This is the honest number."""
    rows = []
    for ct in thresholds:
        idx = cfg.taxon_class_index[ct.label]
        scores = probs_te[:, idx]
        pos = y_te == idx
        sel = scores >= ct.threshold
        tp = int((pos & sel).sum())
        fp = int((~pos & sel).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(int(pos.sum()), 1)
        lo, hi = wilson_interval(tp, max(tp + fp, 1))
        rows.append(
            {
                "label": ct.label,
                "common_name": ct.common_name,
                "threshold": round(ct.threshold, 3),
                "test_precision": round(prec, 4),
                "test_precision_ci": [round(lo, 4), round(hi, 4)],
                "test_recall": round(rec, 4),
                "n_test": int(pos.sum()),
                "fired": int(sel.sum()),
                "target_achievable": ct.achievable,
            }
        )
    return rows


# --- false triggers -----------------------------------------------------------


def _visits(ids: np.ndarray, min_frames: int = 2) -> list[np.ndarray]:
    """Group frame indices into visits by observation id."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, o in enumerate(ids):
        if o:
            by[str(o)].append(i)
    return [np.array(v) for v in by.values() if len(v) >= min_frames]


def false_trigger_curve(
    cfg: Config,
    novelty_scorer,
    feats_va: np.ndarray,
    logits_va: np.ndarray,
    feats_te: np.ndarray,
    logits_te: np.ndarray,
    probs_te: np.ndarray,
    y_te: np.ndarray,
    obs_te: np.ndarray,
    feats_ood: np.ndarray,
    logits_ood: np.ndarray,
    probs_ood: np.ndarray,
    obs_ood: np.ndarray,
    class_thresholds: dict[str, float],
    fars=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
) -> list[dict]:
    """Sweep the novelty false-alarm rate; report frame- AND visit-level triggers.

    A trigger means: not flagged unknown, AND some Tier A species clears its
    per-class threshold. That is the condition under which video is committed.
    """
    tier_a_idx = {
        cfg.taxon_class_index[s.slug]: s.slug
        for s in cfg.species_by_tier("A")
        if s.slug in cfg.taxon_class_index
    }

    def frame_triggers(probs, unknown_mask):
        fire = np.zeros(len(probs), bool)
        for idx, slug in tier_a_idx.items():
            thr = class_thresholds.get(slug, 0.55)
            fire |= probs[:, idx] >= thr
        return fire & ~unknown_mask

    te_visits = _visits(obs_te)
    ood_visits = _visits(obs_ood)

    rows = []
    for far in fars:
        thr = novelty_scorer.calibrate(feats_va, logits_va, target_fpr=far)
        unk_te = novelty_scorer.score(feats_te, logits_te) > thr
        unk_ood = novelty_scorer.score(feats_ood, logits_ood) > thr

        fire_te = frame_triggers(probs_te, unk_te)
        fire_ood = frame_triggers(probs_ood, unk_ood)

        # Visit level: majority-unknown suppresses the visit, otherwise a visit
        # fires if any sampled frame fires (matching the capture logic).
        def visit_fire(visits, fire, unk):
            out = []
            for v in visits:
                if unk[v].mean() > 0.5:
                    out.append(False)
                else:
                    out.append(bool(fire[v].any()))
            return np.array(out)

        vt, vo = visit_fire(te_visits, fire_te, unk_te), visit_fire(ood_visits, fire_ood, unk_ood)

        # Of bird frames that fired, how many named the right species?
        pred = probs_te.argmax(1)
        correct = float((pred[fire_te] == y_te[fire_te]).mean()) if fire_te.sum() else 0.0

        rows.append(
            {
                "target_far": far,
                "novelty_threshold": round(float(thr), 4),
                "frame_bird_trigger": round(float(fire_te.mean()), 4),
                "frame_ood_trigger": round(float(fire_ood.mean()), 4),
                "visit_bird_trigger": round(float(vt.mean()), 4),
                "visit_ood_trigger": round(float(vo.mean()), 4),
                "precision_of_fired_frames": round(correct, 4),
                "n_bird_visits": len(te_visits),
                "n_ood_visits": len(ood_visits),
            }
        )
    return rows


def write_thresholds_to_config(cfg: Config, rows: list[dict]) -> None:
    """Write fitted per-class thresholds into config/taxonomy.yaml.

    Text-level edit of the existing block so comments elsewhere survive.
    """
    path = cfg.root / "config" / "taxonomy.yaml"
    text = path.read_text(encoding="utf-8")
    block = ["  per_class_thresholds:"]
    for r in sorted(rows, key=lambda x: x["label"]):
        block.append(
            f"    {r['label']}: {r['threshold']}"
            f"   # test precision {r['test_precision']:.3f}, recall {r['test_recall']:.3f}"
        )
    new = "\n".join(block)
    old_marker = "  per_class_thresholds: {}"
    if old_marker in text:
        text = text.replace(old_marker, new)
    else:
        import re

        text = re.sub(r"  per_class_thresholds:\n(?:    .*\n)*", new + "\n", text, count=1)
    path.write_text(text, encoding="utf-8")
    logger.info("wrote %d per-class thresholds to %s", len(rows), path)


# --- driver -------------------------------------------------------------------


def _calibrate_and_measure(
    cfg: Config,
    X: np.ndarray,
    logits: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    obs: np.ndarray,
    ood_feats: np.ndarray,
    ood_logits: np.ndarray,
    ood_obs: np.ndarray,
    target_precision: float,
    source: str,
    out_name: str,
):
    """Temperature, per-class thresholds, novelty scorer and the trigger curve.

    Shared by both entry points so the frozen-probe path and the fine-tuned
    checkpoint path cannot drift apart in how they calibrate or measure.
    """
    import torch
    import torch.nn as nn

    from birdcam.models.novelty import KNNScorer

    ce = nn.CrossEntropyLoss()
    tr, va, te = split == "train", split == "val", split == "test"
    lva, lte, lood = logits[va], logits[te], ood_logits

    # Temperature on val -- calibrated probabilities are the point of this whole
    # exercise, since the thresholds below are read off them.
    T = torch.ones(1, requires_grad=True)
    o = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
    tv, yv = torch.tensor(lva), torch.tensor(y[va])

    def closure():
        o.zero_grad()
        loss = ce(tv / T.clamp(min=1e-2), yv)
        loss.backward()
        return loss

    o.step(closure)
    temp = float(T.detach().clamp(min=1e-2))

    def sm(z):
        z = np.asarray(z, dtype=np.float64) / temp
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    p_va, p_te, p_ood = sm(lva), sm(lte), sm(lood)

    fitted = fit_class_thresholds(p_va, y[va], cfg, target_precision)
    rows = evaluate_thresholds(p_te, y[te], cfg, fitted)

    scorer = KNNScorer(k=10, max_reference=1000).fit(X[tr], y[tr])
    curve = false_trigger_curve(
        cfg,
        scorer,
        X[va],
        lva,
        X[te],
        lte,
        p_te,
        y[te],
        obs[te],
        ood_feats,
        lood,
        p_ood,
        ood_obs,
        {r["label"]: r["threshold"] for r in rows},
    )

    out = {
        "source": source,
        "temperature": round(temp, 3),
        "target_precision": target_precision,
        "per_class": rows,
        "false_trigger_curve": curve,
    }
    # Keyed by source: the frozen-probe result is the only "before" we have, and
    # overwriting it would destroy the comparison this work exists to make.
    path = cfg.path("reports_dir") / out_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return out


def run(cfg: Config, feature_file: str, target_precision: float = 0.90, epochs: int = 200):
    """Frozen-backbone path: train a linear probe over cached sweep embeddings.

    This does NOT evaluate a fine-tuned checkpoint. Use `run_from_extraction`
    for that.
    """
    import torch
    import torch.nn as nn

    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest

    emb = cfg.path("embeddings_dir") / "sweep"
    X = np.load(emb / feature_file)
    ood = np.load(emb / f"ood_{feature_file}")

    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
        ood_rows = list(m.iter_rows("tier='OOD' AND status='downloaded'"))
    if len(X) != len(items):
        raise RuntimeError(f"misalignment: {len(X)} features vs {len(items)} items")

    split = np.array([i.split for i in items])
    y = np.array([i.taxon_index for i in items])
    obs = np.array([i.observation_id or "" for i in items])
    tr = split == "train"

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xz = (X - mu) / sd
    ood_z = (ood - mu) / sd

    head = nn.Linear(X.shape[1], len(cfg.taxon_classes))
    opt = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    xt, yt = torch.tensor(Xz[tr], dtype=torch.float32), torch.tensor(y[tr])
    for _ in range(epochs):
        opt.zero_grad()
        ce(head(xt), yt).backward()
        opt.step()
    head.eval()

    def lg(a):
        with torch.no_grad():
            return head(torch.tensor(a, dtype=torch.float32)).numpy()

    return _calibrate_and_measure(
        cfg,
        X,
        lg(Xz),
        y,
        split,
        obs,
        ood,
        lg(ood_z),
        np.array([r["observation_id"] or "" for r in ood_rows]),
        target_precision,
        source=f"frozen probe over {feature_file}",
        out_name="operating_points.json",
    )


def run_from_extraction(
    cfg: Config, checkpoint_name: str = "student_best.pt", target_precision: float = 0.90
):
    """Fine-tuned path: use the checkpoint's own features and logits.

    No probe is trained here. The logits are the deployed model's logits, so
    temperature, thresholds and novelty all describe the model that will ship.
    """
    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest
    from birdcam.eval.extract import load_extraction, ood_items

    stem = checkpoint_name.replace(".pt", "")
    emb = cfg.path("embeddings_dir") / "finetuned"

    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
        ood_rows = list(m.iter_rows("tier='OOD' AND status='downloaded'"))
        ood_it = ood_items(cfg, m)

    idx = load_extraction(cfg, emb / f"{stem}_id.npz", [i.image_id for i in items])
    ood = load_extraction(cfg, emb / f"{stem}_ood.npz", [i.image_id for i in ood_it])
    logger.info("using %s (checkpoint sha %s)", idx.checkpoint, idx.checkpoint_sha[:12])

    return _calibrate_and_measure(
        cfg,
        idx.features,
        idx.taxon_logits,
        np.array([i.taxon_index for i in items]),
        np.array([i.split for i in items]),
        np.array([i.observation_id or "" for i in items]),
        ood.features,
        ood.taxon_logits,
        np.array([r["observation_id"] or "" for r in ood_rows]),
        target_precision,
        source=f"fine-tuned checkpoint {checkpoint_name} @ {idx.checkpoint_sha[:12]}",
        out_name="operating_points_finetuned.json",
    )


def print_report(res: dict) -> None:
    print(f"\ntemperature T={res['temperature']}  (probabilities are calibrated)")
    print(f"\nPER-CLASS THRESHOLDS -- fitted on val for >={res['target_precision']:.0%} precision,")
    print("evaluated on test. 'fired' = frames clearing the threshold.\n")
    hdr = f"{'Tier A species':<32}{'thr':>6}{'precision':>21}{'recall':>8}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(res["per_class"], key=lambda x: -x["test_precision"]):
        ci = f"[{r['test_precision_ci'][0]:.2f}-{r['test_precision_ci'][1]:.2f}]"
        flag = "" if r["target_achievable"] else "  UNREACHABLE"
        print(
            f"{r['common_name']:<32}{r['threshold']:>6.2f}"
            f"{r['test_precision']:>9.3f} {ci:>11}{r['test_recall']:>8.3f}{r['n_test']:>6}{flag}"
        )

    print("\nFALSE TRIGGERS vs NOVELTY SENSITIVITY")
    print("A trigger = not flagged unknown AND some Tier A class clears its threshold.")
    print("Visit level is the number that matters: a visit is many frames and the")
    print("track vote decides, so frame rates overstate the problem.\n")
    c = res["false_trigger_curve"]
    hdr2 = (
        f"{'novelty FAR':>12}{'bird frames':>13}{'OOD frames':>12}"
        f"{'BIRD VISITS':>13}{'OOD VISITS':>12}{'prec of fired':>15}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for r in c:
        print(
            f"{r['target_far']:>12.0%}{r['frame_bird_trigger']:>13.3f}"
            f"{r['frame_ood_trigger']:>12.3f}{r['visit_bird_trigger']:>13.3f}"
            f"{r['visit_ood_trigger']:>12.3f}{r['precision_of_fired_frames']:>15.3f}"
        )
    print(f"\n({c[0]['n_bird_visits']} bird visits, {c[0]['n_ood_visits']} OOD visits)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(
        description="Fit confidence thresholds and measure false triggers."
    )
    ap.add_argument("--features", default="tf_efficientnetv2_b0.in1k_18146.npy")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "evaluate a fine-tuned checkpoint (e.g. student_best.pt) using its own "
            "features and logits. Requires birdcam.eval.extract to have been run. "
            "Without this flag a linear probe is trained over frozen sweep "
            "embeddings, which does NOT measure any fine-tuned model."
        ),
    )
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument(
        "--write-config",
        action="store_true",
        help="write fitted thresholds into config/taxonomy.yaml",
    )
    args = ap.parse_args()

    cfg = load_config()
    if args.checkpoint:
        res = run_from_extraction(cfg, args.checkpoint, args.target_precision)
    else:
        res = run(cfg, args.features, args.target_precision)
    print(f"\nsource: {res['source']}")
    print_report(res)
    if args.write_config:
        write_thresholds_to_config(cfg, res["per_class"])
        print("\nwrote per_class_thresholds to config/taxonomy.yaml")


if __name__ == "__main__":
    main()
