"""Prepare an isolated INT8 capture calibration candidate; never edit live config.

Fits on validation only using capture preprocessing and inference's own logit
transform. Reported validation results are fitting diagnostics, not held-out
accuracy. Replay and field validation are required before deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from birdcam.config import load_config
from birdcam.data.dataset import load_labelled
from birdcam.data.manifest import open_manifest
from birdcam.inference import Classifier
from birdcam.models.novelty import EnergyScorer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_operating_points(cfg, logits, items, prior, target_precision=0.8):
    """Conditional ID calibration; report suppressed empty scenes separately."""
    y = np.array([i.taxon_index for i in items])
    base = Classifier(cfg, range_prior=prior)
    adjusted = base.probability_logits(logits)
    eligible = ~np.isin(y, base._suppressed)
    z, yy = adjusted[eligible], y[eligible]
    if not len(z):
        raise ValueError("no eligible calibration rows")

    def nll(log_t):
        a = z / np.exp(log_t)
        return float(np.mean(logsumexp(a, axis=1) - a[np.arange(len(a)), yy]))

    fit = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    if not fit.success:
        raise RuntimeError("temperature fit failed")
    temperature = float(np.exp(fit.x))
    scorer = EnergyScorer()
    energy = scorer.score(None, logits)
    birds = set(base._species_idx.values())
    field_birds = np.array([i.source == "field" and i.taxon_index in birds for i in items])
    if not field_birds.any():
        raise ValueError("no field validation birds for energy threshold")
    scorer.threshold = float(np.quantile(energy[field_birds], 0.9))
    p = softmax(adjusted / temperature, axis=1)
    winner = p.argmax(axis=1)
    accepted = energy <= scorer.threshold
    thresholds = []
    for sp in cfg.species_by_tier("A"):
        idx = cfg.taxon_class_index[sp.slug]
        row = dict(
            label=sp.slug,
            common_name=sp.common_name,
            threshold=1.0,
            target_achievable=False,
            validation_predictions=0,
            validation_precision=None,
        )
        for t in np.linspace(0.05, 0.99, 95):
            sel = (winner == idx) & accepted & (p[:, idx] >= t)
            if sel.sum() >= 5 and (y[sel] == idx).mean() >= target_precision:
                row.update(
                    threshold=float(t),
                    target_achievable=True,
                    validation_predictions=int(sel.sum()),
                    validation_precision=float((y[sel] == idx).mean()),
                )
                break
        thresholds.append(row)
    # Evaluate actual frame decisions; visit voting is exercised by Pi replay.
    cfg.taxonomy_cfg["rollup"]["per_class_thresholds"] = {
        r["label"]: r["threshold"] for r in thresholds
    }
    decider = Classifier(cfg, novelty_scorer=scorer, temperature=temperature, range_prior=prior)
    decisions = [decider.decide(row_logits) for row_logits in logits]
    by_source = {}
    for source in sorted({i.source for i in items}):
        sel = np.array([i.source == source for i in items])
        by_source[source] = dict(
            rows=int(sel.sum()),
            predicted_unknown=sum(
                d.is_unknown for d, keep in zip(decisions, sel, strict=True) if keep
            ),
            would_publish=sum(
                d.should_record for d, keep in zip(decisions, sel, strict=True) if keep
            ),
        )
    empty = np.array([i.taxon_label == "empty_feeder" for i in items])
    return dict(
        temperature=temperature,
        target_precision=target_precision,
        per_class=thresholds,
        energy_threshold=scorer.threshold,
        energy_fit_field_bird_rows=int(field_birds.sum()),
        calibration_rows=int(eligible.sum()),
        suppressed_rows=int((~eligible).sum()),
        conditional_nll_before=nll(0),
        conditional_nll_after=nll(fit.x),
        validation_diagnostics=by_source,
        empty_frames_would_publish_without_geometric_gate=sum(
            d.should_record for d, keep in zip(decisions, empty, strict=True) if keep
        ),
        empty_validation_frames=int(empty.sum()),
        caveat="Validation fitting diagnostics only; no held-out or live performance claim. "
        "Empty reference gate bypassed. Per-class precision does not guarantee genus precision.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--site", type=Path, default=Path("ml/config/sites/rondebosch.yaml"))
    args = ap.parse_args()
    import onnxruntime as ort

    from capture.classifier import load_range_prior, preprocess_frame

    cfg = load_config()
    meta = json.loads(args.onnx.with_suffix(".json").read_text())
    graph_sha = digest(args.onnx)
    if meta["taxon_classes"] != list(cfg.taxon_classes):
        raise ValueError("graph class order mismatch")
    if meta["quantisation"]["onnx_sha256"] != graph_sha:
        raise ValueError("graph hash mismatch")
    with open_manifest(cfg.path("manifest_db")) as m:
        items = [i for i in load_labelled(cfg, m, include_field=True) if i.split == "val"]
    args.out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        onnx_sha256=graph_sha,
        classes=list(cfg.taxon_classes),
        ids=[i.image_id for i in items],
        image_sha256=[digest(i.path) for i in items],
        preprocessing_sha256=digest(Path("capture/classifier.py")),
    )
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = args.out / "validation_logits.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["fingerprint"]) != fingerprint:
                raise ValueError("existing output cache differs; choose a new output directory")
            logits = data["logits"]
    else:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        sess = ort.InferenceSession(str(args.onnx), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        arrays = []
        for k in range(0, len(items), 16):
            batch = np.stack(
                [preprocess_frame(i.path, meta["image_size"]) for i in items[k : k + 16]]
            )
            arrays.append(sess.run(["taxon_logits"], {name: batch})[0])
            if k % 256 == 0:
                print(f"capture preprocessing: {k}/{len(items)}", flush=True)
        logits = np.concatenate(arrays)
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite model outputs")
        temp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp, logits=logits, fingerprint=fingerprint, ids=np.array(identity["ids"], dtype=str)
        )
        temp.replace(cache)
    prior = load_range_prior(args.site)
    report = fit_operating_points(cfg, logits, items, prior)
    report.update(
        source=f"INT8 checkpoint {meta['checkpoint']} @ {meta['checkpoint_sha'][:12]}",
        checkpoint_sha=meta["checkpoint_sha"],
        onnx_sha256=graph_sha,
        validation_fingerprint=fingerprint,
        site_sha256=digest(args.site),
        suppressed_at_inference=cfg.taxonomy_cfg["taxon_head"]["suppressed_at_inference"],
    )
    (args.out / "operating_points.json").write_text(json.dumps(report, indent=2))
    (args.out / "taxonomy.yaml").write_text(yaml.safe_dump(cfg.taxonomy_cfg, sort_keys=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
