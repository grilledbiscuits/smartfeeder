"""Report field accuracy separately for deployment framing and pillarboxed frames.

    python -m birdcam.eval.framing_report

Every field metric quoted before 2026-09-21 pooled two populations: frames shot
in the feeder's landscape framing, and frames that are portrait video padded
into a landscape frame with 68% black bars. The Pi's camera can only produce the
first, so a pooled number answers a question nobody asked. This splits them.

Reads logits that already exist rather than running the model:

* `ml/data/embeddings/finetuned/student_best_id.npz` -- the FP32 checkpoint over
  every split, written by `eval.extract`.
* `ml/reports/<run>/validation_logits.npz` -- the INT8 graph over validation with
  capture preprocessing, written by `eval.calibrate_capture`. This is the model
  and the input path the Pi actually uses, so it is the one to quote.

Accuracy here is plain argmax over the taxon head with `empty_feeder` suppressed,
which is NOT the deployed decision: the service also applies a site prior,
temperature, the novelty gate, rollup and per-class thresholds, and votes over a
clip. Treat this as a framing comparison, not a deployment claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from birdcam.config import load_config
from birdcam.data.dataset import load_labelled
from birdcam.data.letterbox import letterboxed_ids
from birdcam.data.manifest import open_manifest
from birdcam.inference import Classifier


def accuracy(logits: np.ndarray, rows, suppressed) -> tuple[float | None, int]:
    if not len(rows):
        return None, 0
    z = logits.copy()
    z[:, suppressed] = -np.inf
    truth = np.array([r.taxon_index for r in rows])
    return float((z.argmax(1) == truth).mean()), len(rows)


def compare(logits, items, barred, suppressed, classes) -> dict:
    """Accuracy on deployment-framed vs pillarboxed field frames."""
    index = {i.image_id: n for n, i in enumerate(items)}

    def group(rows):
        if not rows:
            return dict(accuracy=None, frames=0)
        acc, n = accuracy(logits[[index[r.image_id] for r in rows]], rows, suppressed)
        return dict(accuracy=None if acc is None else round(acc, 4), frames=n)

    field = [i for i in items if i.source == "field" and i.taxon_label != "empty_feeder"]
    out: dict = {}
    for split in sorted({i.split for i in field}):
        rows = [i for i in field if i.split == split]
        clean = [i for i in rows if i.image_id not in barred]
        bars = [i for i in rows if i.image_id in barred]
        per_class = {}
        for label in sorted({i.taxon_label for i in rows}):
            per_class[label] = dict(
                deployment_framing=group([i for i in clean if i.taxon_label == label]),
                pillarboxed=group([i for i in bars if i.taxon_label == label]),
            )
        out[split] = dict(
            deployment_framing=group(clean),
            pillarboxed=group(bars),
            per_class=per_class,
        )
    return out


def table(name: str, report: dict, common: dict) -> list[str]:
    lines = [
        f"### {name}",
        "",
        "| Split / class | Deployment framing | Pillarboxed |",
        "|---|---|---|",
    ]

    def cell(g):
        return "—" if g["accuracy"] is None else f"{g['accuracy']:.3f} (n={g['frames']})"

    for split, data in report.items():
        lines.append(
            f"| **{split}, all field birds** | **{cell(data['deployment_framing'])}** "
            f"| **{cell(data['pillarboxed'])}** |"
        )
        for label, groups in data["per_class"].items():
            lines.append(
                f"| {common.get(label, label)} | {cell(groups['deployment_framing'])} "
                f"| {cell(groups['pillarboxed'])} |"
            )
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction", type=Path, default=Path("ml/data/embeddings/finetuned"))
    ap.add_argument("--int8-logits", type=Path, default=Path("ml/reports/clean-20260921"))
    ap.add_argument("--out", type=Path, default=Path("ml/reports/framing"))
    args = ap.parse_args()

    cfg = load_config()
    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m, include_field=True)
    barred = letterboxed_ids(items, cache_path=Path("ml/data/letterboxed.json"))
    suppressed = Classifier(cfg)._suppressed
    common = {s.slug: s.common_name for s in cfg.species}
    report: dict = {
        "suppressed_at_inference": [cfg.taxon_classes[i] for i in suppressed],
        "pillarboxed_frames": len(barred),
        "field_frames": sum(1 for i in items if i.source == "field"),
        "caveat": "Argmax over the taxon head only: no prior, temperature, novelty gate, "
        "rollup, per-class thresholds or clip vote. Framing comparison, not deployment "
        "performance.",
    }

    fp32 = args.extraction / "student_best_id.npz"
    with np.load(fp32, allow_pickle=True) as data:
        order = {str(v): n for n, v in enumerate(data["image_ids"])}
        rows = [i for i in items if i.image_id in order]
        logits = data["taxon_logits"][[order[i.image_id] for i in rows]]
        assert list(data["taxon_classes"]) == list(cfg.taxon_classes), "class order moved"
        report["fp32_checkpoint"] = compare(logits, rows, barred, suppressed, cfg.taxon_classes)
        report["fp32_checkpoint_sha"] = str(data["checkpoint_sha"])

    int8 = args.int8_logits / "validation_logits.npz"
    if int8.is_file():
        with np.load(int8, allow_pickle=False) as data:
            ids = [str(v) for v in data["ids"]]
            by_id = {i.image_id: i for i in items}
            rows = [by_id[image_id] for image_id in ids if image_id in by_id]
            report["int8_capture_path"] = compare(
                data["logits"], rows, barred, suppressed, cfg.taxon_classes
            )
            report["int8_source"] = str(args.int8_logits)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "framing_accuracy.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Field accuracy by framing",
        "",
        f"{report['pillarboxed_frames']} of {report['field_frames']} field frames are "
        "portrait video padded into a landscape frame (68% black bars). The deployment "
        "camera cannot produce them, so they are reported separately rather than pooled.",
        "",
        f"*{report['caveat']}*",
        "",
    ]
    md += table("FP32 checkpoint (all splits)", report["fp32_checkpoint"], common)
    if "int8_capture_path" in report:
        md += table(
            "INT8 graph with capture preprocessing — the deployed path (validation)",
            report["int8_capture_path"],
            common,
        )
    (args.out / "framing_accuracy.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
