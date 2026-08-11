"""Sweep runner: backbone and head configurations vs accuracy AND compute cost.

Answers two questions at once:

1. Which backbone gives the best Tier A accuracy?
2. What does it cost to run once trained?

## On measuring cost

Cost is reported as **parameter count and multiply-accumulates (MACs)** at the
deployment input resolution. Both are properties of the architecture and are
identical on any machine.

Wall-clock latency is deliberately NOT reported. This development machine is an
x86 laptop with no accelerator; the deployment target is a Raspberry Pi 5 with a
Hailo-8L. A latency number measured here would not transfer, and quoting one
would invite exactly the wrong decision. MACs are the honest proxy until the
model is compiled and measured on the real hardware.

## On architecture eligibility

`hailo_ok` marks whether a backbone is a plain CNN. The Hailo Dataflow Compiler
handles standard convolutional operations well but has limited and awkward
support for transformer and ConvNeXt blocks (LayerNorm, GELU, attention). Models
marked False may still be useful as an accuracy *ceiling* -- what a bigger model
could reach -- but must not be selected as the student.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass

import numpy as np

from birdcam.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class BackboneSpec:
    name: str
    image_size: int = 224
    batch_size: int = 16
    # False => usable as an accuracy ceiling only, never as the deployed student.
    hailo_ok: bool = True
    note: str = ""


# Deployment-realistic candidates. All CNNs except the explicitly marked ceiling
# references.
CANDIDATES: list[BackboneSpec] = [
    BackboneSpec("mobilenetv3_large_100.ra_in1k", note="current student default"),
    BackboneSpec("mobilenetv4_conv_small.e2400_r224_in1k", note="newest small CNN"),
    BackboneSpec("mobilenetv4_conv_medium.e500_r224_in1k", note="mid MobileNetV4"),
    BackboneSpec("efficientnet_lite0.ra_in1k", note="designed for edge INT8"),
    BackboneSpec("efficientnet_b0.ra4_e3600_r224_in1k"),
    BackboneSpec("tf_efficientnetv2_b0.in1k"),
    BackboneSpec("resnet50.a1_in1k", batch_size=8, note="heavy CNN reference"),
    BackboneSpec(
        "convnext_tiny.in12k_ft_in1k",
        batch_size=8,
        hailo_ok=False,
        note="ACCURACY CEILING ONLY -- ConvNeXt blocks compile poorly on Hailo",
    ),
]


@dataclass
class SweepRow:
    backbone: str
    hailo_ok: bool
    params_m: float
    macs_g: float
    feature_dim: int
    taxon_acc: float
    taxon_ci_low: float
    taxon_ci_high: float
    sex_acc: float
    tier_a_mean_recall: float
    female_mean_recall: float
    extract_seconds: float
    note: str = ""


def _probe_feature_dim(model, image_size: int) -> int:
    """Actual embedding width, measured by a forward pass.

    `model.num_features` is unreliable across architectures (see call site), and
    a mismatch corrupts the feature matrix silently rather than raising.
    """
    import torch

    with torch.inference_mode():
        out = model(torch.zeros(1, 3, image_size, image_size))
    return int(out.shape[1])


def compute_cost(model, image_size: int) -> tuple[float, float]:
    """Return (params in millions, MACs in billions) at the given resolution.

    FlopCounterMode reports FLOPs counting a multiply-add as 2; MACs are half
    that, which is the convention accelerator datasheets use.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(1, 3, image_size, image_size)
    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        model(x)
    macs = counter.get_total_flops() / 2 / 1e9
    return params, macs


def run_sweep(cfg: Config, specs: list[BackboneSpec] | None = None, epochs: int = 200):
    """Extract features with each backbone, train heads, tabulate accuracy vs cost."""
    import timm
    import torch

    from birdcam.data.dataset import ImageDataset, load_labelled
    from birdcam.data.manifest import open_manifest
    from birdcam.train.train_head import fit_and_eval
    from birdcam.utils.runtime import setup_torch

    setup_torch(cfg)
    specs = specs or CANDIDATES

    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
    if not items:
        raise RuntimeError("no labelled images; run the fetcher and preprocess first")
    logger.info("sweep over %d backbones, %d images", len(specs), len(items))

    rows: list[SweepRow] = []
    for spec in specs:
        logger.info("--- %s ---", spec.name)
        try:
            model = timm.create_model(spec.name, pretrained=True, num_classes=0)
        except Exception as exc:  # noqa: BLE001
            logger.error("skipping %s: %s", spec.name, exc)
            continue
        model.eval()

        params_m, macs_g = compute_cost(model, spec.image_size)
        # Probe the real output width rather than trusting `num_features`.
        # They disagree on several architectures -- MobileNetV3 reports 960 but
        # emits 1280, because timm's conv head sits after the layer that
        # `num_features` describes. Trusting the attribute silently corrupts the
        # feature matrix.
        dim = _probe_feature_dim(model, spec.image_size)

        # Match the transform to this backbone's own pretraining statistics.
        data_cfg = timm.data.resolve_data_config({}, model=model)
        data_cfg["input_size"] = (3, spec.image_size, spec.image_size)
        transform = timm.data.create_transform(**data_cfg, is_training=False)

        # Cache per backbone. Extraction is the expensive part of the sweep
        # (minutes each); a failure downstream must not cost it again.
        cache_dir = cfg.path("embeddings_dir") / "sweep"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{spec.name.replace('/', '_')}_{len(items)}.npy"

        if cache_path.is_file():
            feats = np.load(cache_path)
            extract_s = 0.0
            logger.info("  reusing cached features %s", cache_path.name)
        else:
            loader = torch.utils.data.DataLoader(
                ImageDataset(items, transform),
                batch_size=spec.batch_size,
                shuffle=False,
                num_workers=cfg.train_cfg["compute"]["dataloader_num_workers"],
            )
            feats = np.zeros((len(items), dim), dtype=np.float32)
            t0 = time.monotonic()
            with torch.inference_mode():
                for batch, idx in loader:
                    feats[idx.numpy()] = model(batch).float().numpy()
            extract_s = time.monotonic() - t0
            np.save(cache_path, feats)
        logger.info(
            "  extracted in %.0fs (%.1f img/s), %.1fM params, %.2fG MACs",
            extract_s,
            len(items) / extract_s,
            params_m,
            macs_g,
        )

        res = fit_and_eval(cfg, feats, items, epochs=epochs)
        rows.append(
            SweepRow(
                backbone=spec.name,
                hailo_ok=spec.hailo_ok,
                params_m=round(params_m, 2),
                macs_g=round(macs_g, 3),
                feature_dim=dim,
                taxon_acc=round(res["taxon_acc"], 4),
                taxon_ci_low=round(res["taxon_ci"][0], 4),
                taxon_ci_high=round(res["taxon_ci"][1], 4),
                sex_acc=round(res["sex_acc"], 4),
                tier_a_mean_recall=round(res["tier_a_mean_recall"], 4),
                female_mean_recall=round(res["female_mean_recall"], 4),
                extract_seconds=round(extract_s, 1),
                note=spec.note,
            )
        )
        del model, feats

    out = cfg.path("reports_dir") / "backbone_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    logger.info("wrote %s", out)
    return rows


def print_table(rows: list[SweepRow]) -> None:
    print(
        "\nCOST IS ARCHITECTURAL (params / MACs). Latency is NOT measured: this is an\n"
        "x86 laptop, the target is a Pi 5 + Hailo-8L, and an off-target latency number\n"
        "would be misleading.\n"
    )
    hdr = (
        f"{'backbone':<42}{'hailo':>6}{'params':>8}{'MACs':>7}"
        f"{'taxon':>8}{'95% CI':>15}{'tierA':>7}{'female':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x.taxon_acc):
        ci = f"{r.taxon_ci_low:.3f}-{r.taxon_ci_high:.3f}"
        mark = "yes" if r.hailo_ok else "NO"
        print(
            f"{r.backbone[:41]:<42}{mark:>6}{r.params_m:>8.1f}{r.macs_g:>7.2f}"
            f"{r.taxon_acc:>8.3f}{ci:>15}{r.tier_a_mean_recall:>7.3f}{r.female_mean_recall:>8.3f}"
        )

    eligible = [r for r in rows if r.hailo_ok]
    if not eligible:
        return
    print("\nPareto frontier (Hailo-eligible only; no cheaper model is also more accurate):")
    for r in sorted(eligible, key=lambda x: x.macs_g):
        if not any(o.macs_g <= r.macs_g and o.taxon_acc > r.taxon_acc for o in eligible):
            print(
                f"  {r.backbone:<45} {r.macs_g:>6.2f}G MACs  "
                f"{r.params_m:>5.1f}M params  taxon {r.taxon_acc:.3f}"
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    import argparse

    ap = argparse.ArgumentParser(description="Sweep backbones for accuracy vs compute cost.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on backbone names")
    args = ap.parse_args()

    cfg = load_config()
    specs = CANDIDATES
    if args.only:
        specs = [s for s in CANDIDATES if any(o in s.name for o in args.only)]
    rows = run_sweep(cfg, specs, epochs=args.epochs)
    print_table(rows)


if __name__ == "__main__":
    main()
