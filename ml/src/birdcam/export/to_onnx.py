"""PyTorch -> ONNX export for the two-head deployment model.

Kept working from early on, because ONNX is the pivot point for every runtime
under consideration: ONNX Runtime on a Pi 4B CPU, and the Hailo Dataflow
Compiler on a Pi 5 + AI HAT+. The board is not yet decided and this file does
not care -- which is the point. A break here blocks deployment either way.

What is deliberately NOT in the graph:

* **Softmax.** Kept outside so calibration and thresholding operate on logits,
  and so the graph is unchanged if the probability treatment changes.
* **Rollup and thresholds.** These are tuned per class from precision-recall
  curves and change without retraining. Baking them in would force a Hailo
  recompile -- slow, and run on a separate x86 toolchain -- every time a
  threshold moves.
* **Normalisation.** Mean/std subtraction stays in the capture application,
  where it can be fused into the camera pipeline.

The exported graph is therefore: preprocessed tensor in, two logit tensors out.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from birdcam.config import Config, load_config

logger = logging.getLogger(__name__)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, matching `eval.extract._sha256_file`.

    Same convention on both sides so a sidecar SHA and a thresholds SHA are
    comparable without anyone having to check how each was computed.
    """
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def export(
    cfg: Config,
    role: str = "student",
    checkpoint: Path | None = None,
    out_path: Path | None = None,
    verify: bool = True,
) -> Path:
    """Export the two-head model to ONNX.

    Exports randomly-initialised weights when no checkpoint is supplied. That is
    useful for validating the graph and measuring its cost, but such a file must
    never be shipped -- the metadata sidecar records which case it was.
    """
    import torch

    from birdcam.models.heads import build_model

    model, backbone_name = build_model(cfg, role)
    trained = False
    checkpoint_sha = ""
    if checkpoint and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        trained = True
        checkpoint_sha = _sha256_file(checkpoint)
        logger.info("loaded weights from %s (sha %s)", checkpoint, checkpoint_sha[:12])
    else:
        logger.warning(
            "NO CHECKPOINT -- exporting randomly-initialised weights. The graph "
            "is valid and its cost is real, but the predictions are noise. Do "
            "not deploy this file."
        )
    model.eval()

    size = cfg.train_cfg["backbone"][role]["image_size"]
    dummy = torch.randn(1, 3, size, size)
    out_path = out_path or (cfg.root / cfg.train_cfg["export"]["onnx_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["taxon_logits", "sex_logits"],
        # Batch is dynamic; spatial dims are fixed. Hailo requires static
        # spatial shapes, and the capture app always feeds a fixed crop size.
        dynamic_axes={
            "image": {0: "batch"},
            "taxon_logits": {0: "batch"},
            "sex_logits": {0: "batch"},
        },
        opset_version=cfg.train_cfg["export"]["onnx_opset"],
        do_constant_folding=True,
    )
    _consolidate_external_data(out_path)
    logger.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Class order is meaningless without the label list, and a mismatch between
    # graph and labels is silent. Ship them together.
    meta = {
        "backbone": backbone_name,
        "role": role,
        "trained": trained,
        # PROVENANCE. A26: a frozen-feature export was paired with thresholds
        # fitted on the fine-tuned checkpoint, and nothing failed -- the labels
        # were simply wrong. The sidecar could not say which model it held,
        # while taxonomy.yaml could. Stamping the checkpoint and its SHA here
        # is what lets a consumer verify the pairing instead of assuming it;
        # the format matches `write_thresholds_to_config`'s `source` so the two
        # strings can be compared directly.
        "training": (
            f"fine-tuned checkpoint {checkpoint.name} @ {checkpoint_sha[:12]}"
            if checkpoint_sha
            else "no checkpoint -- randomly-initialised weights"
        ),
        "checkpoint": checkpoint.name if checkpoint_sha else None,
        "checkpoint_sha": checkpoint_sha,
        "image_size": size,
        "opset": cfg.train_cfg["export"]["onnx_opset"],
        "taxon_classes": cfg.taxon_classes,
        "sex_classes": cfg.sex_classes,
        "partial_label_groups": cfg.partial_label_groups,
        "notes": (
            "Logits only. Softmax, rollup and thresholding are applied by the "
            "caller. Input expects ImageNet-normalised NCHW float32."
        ),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verify:
        verify_export(out_path, size, len(cfg.taxon_classes), len(cfg.sex_classes))
    return out_path


def export_from_frozen_head(
    cfg: Config, feature_file: str, out_path: Path | None = None, epochs: int = 200
) -> Path:
    """Export a genuinely trained model built from the fast-loop head.

    Phase 6 (`train_full.py`) does not exist yet, so there is no fine-tuned
    checkpoint. But the fast loop DOES produce real trained heads on frozen
    features, and backbone + that head is a real, working classifier -- exactly
    the thing the frozen-feature metrics describe.

    Exporting random weights and quantising them measures nothing: noise
    quantises to noise. This gives quantisation something real to degrade.

    The feature standardisation (subtract mu, divide by sd) is FOLDED into the
    linear layer rather than shipped as a separate step:

        W((x - mu) / sd) + b  ==  (W / sd) x + (b - W (mu / sd))

    so the exported graph needs no preprocessing beyond the usual image
    normalisation.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest
    from birdcam.models.backbone import load_backbone
    from birdcam.models.heads import TwoHeadNet, masked_partial_label_loss

    X = np.load(cfg.path("embeddings_dir") / "sweep" / feature_file)
    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
    if len(X) != len(items):
        raise RuntimeError(f"misalignment: {len(X)} features vs {len(items)} items")

    split = np.array([i.split for i in items])
    y = np.array([i.taxon_index for i in items])
    masks = np.stack([i.sex_mask for i in items])
    tr = split == "train"

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y[tr])
    mt = torch.tensor(masks[tr], dtype=torch.float32)

    taxon = nn.Linear(X.shape[1], len(cfg.taxon_classes))
    sex = nn.Linear(X.shape[1], len(cfg.sex_classes))
    opt = torch.optim.AdamW(
        list(taxon.parameters()) + list(sex.parameters()), lr=0.01, weight_decay=1e-4
    )
    ce = nn.CrossEntropyLoss()
    sex_w = cfg.train_cfg["train"]["loss"]["sex_weight"]
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(taxon(xt), yt) + sex_w * masked_partial_label_loss(sex(xt), mt)
        loss.backward()
        opt.step()
    logger.info("trained heads on frozen features (%d train images)", int(tr.sum()))

    # Fold standardisation into the linear layers.
    scale = torch.tensor(1.0 / sd, dtype=torch.float32)
    shift = torch.tensor(mu / sd, dtype=torch.float32)
    for layer in (taxon, sex):
        with torch.no_grad():
            layer.bias -= layer.weight @ shift
            layer.weight *= scale

    role = "student"
    backbone, name = load_backbone(cfg, role, num_classes=0)
    model = TwoHeadNet(backbone, X.shape[1], len(cfg.taxon_classes), len(cfg.sex_classes),
                       dropout=0.0)
    model.taxon_head, model.sex_head = taxon, sex
    model.eval()

    size = cfg.train_cfg["backbone"][role]["image_size"]
    out_path = out_path or (cfg.root / cfg.train_cfg["export"]["onnx_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 3, size, size),
        str(out_path),
        input_names=["image"],
        output_names=["taxon_logits", "sex_logits"],
        dynamic_axes={"image": {0: "batch"}, "taxon_logits": {0: "batch"},
                      "sex_logits": {0: "batch"}},
        opset_version=cfg.train_cfg["export"]["onnx_opset"],
        do_constant_folding=True,
    )
    _consolidate_external_data(out_path)

    meta = {
        "backbone": name,
        "role": role,
        "trained": True,
        "training": "frozen-feature linear heads, standardisation folded in",
        "feature_file": feature_file,
        "image_size": size,
        "opset": cfg.train_cfg["export"]["onnx_opset"],
        "taxon_classes": cfg.taxon_classes,
        "sex_classes": cfg.sex_classes,
        "partial_label_groups": cfg.partial_label_groups,
        "notes": (
            "Logits only. Softmax, range prior, temperature, rollup and "
            "thresholding are applied by the caller."
        ),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    verify_export(out_path, size, len(cfg.taxon_classes), len(cfg.sex_classes))
    logger.info("exported TRAINED model to %s", out_path)
    return out_path


def _consolidate_external_data(path: Path) -> None:
    """Fold external weight files back into a single .onnx.

    torch's exporter writes weights to a sidecar `.onnx.data` by default. That
    is sensible for multi-GB models and a nuisance for a 4M-parameter one: the
    Hailo parser runs on a separate x86 workstation, and a two-file artefact is
    one more thing to copy wrong. Student models here are tens of MB, so a
    single self-contained file is strictly easier to hand off.
    """
    try:
        import onnx
    except ImportError:
        logger.warning("onnx not installed; leaving external data as-is")
        return

    sidecar = path.with_suffix(path.suffix + ".data")
    if not sidecar.is_file():
        return
    model = onnx.load(str(path))  # pulls in the external tensors
    onnx.save_model(model, str(path), save_as_external_data=False)
    sidecar.unlink()
    logger.info("consolidated external weights into a single file")


def verify_export(path: Path, size: int, n_taxon: int, n_sex: int) -> None:
    """Load the graph and confirm it runs and has the expected output shapes."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnx/onnxruntime not installed; skipping verification (uv sync --extra export)"
        )
        return

    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, size, size).astype(np.float32)
    taxon, sex = sess.run(None, {"image": x})
    assert taxon.shape == (2, n_taxon), f"taxon shape {taxon.shape} != (2, {n_taxon})"
    assert sex.shape == (2, n_sex), f"sex shape {sex.shape} != (2, {n_sex})"
    logger.info("verified: taxon%s sex%s", taxon.shape, sex.shape)


DEPLOY_NOTE = """\
# Deployment notes -- Raspberry Pi 4B and Pi 5

The board is not yet decided. The exported ONNX runs on both; what differs is
where it runs and how fast. Nothing in the model or this repo needs to change to
switch between them -- this note exists so the decision can be made on facts.

## Pi 5 + AI HAT+ (Hailo-8L)

* The AI HAT+ connects over the Pi 5's **PCIe port and requires a Pi 5**. It is
  not compatible with a Pi 4.
* Inference moves off the CPU entirely. 0.72 GMACs is a rounding error against
  13 TOPS, so the classifier stops being the bottleneck.
* Requires the Hailo Dataflow Compiler: an x86-64 Linux toolchain, separate
  account, run on a workstation. Outline (NOT run or verified here):

      hailo parser onnx birdcam_student.onnx --hw-arch hailo8l
      hailo optimize birdcam_student.har --calib-set-path calib/ --hw-arch hailo8l
      hailo compiler birdcam_student_optimized.har --hw-arch hailo8l

* Keep the student a plain CNN. ViT and ConvNeXt blocks (LayerNorm, GELU,
  attention) compile poorly or fall back to CPU for large subgraphs.
* **The Pi 5 has no hardware H.264 encoder** -- it was removed from the BCM2712.
  Video encoding falls to software (libav), consuming CPU that the accelerator
  was supposed to free up.

## Pi 4B 8GB (CPU only)

* No accelerator option. Inference runs on 4x Cortex-A72 @ 1.5GHz with NEON.
* INT8 quantisation becomes mandatory rather than an optimisation. Prefer ONNX
  Runtime with the XNNPACK execution provider, which has NEON INT8 kernels:

      onnxruntime.InferenceSession(
          "birdcam_student.onnx",
          providers=["XnnpackExecutionProvider", "CPUExecutionProvider"],
      )

* Classify SAMPLED frames only, never every frame. A visit lasting seconds
  yields plenty; the track vote does the rest.
* **The Pi 4 retains the hardware H.264 encoder.** `CircularOutput` pre-roll
  costs almost no CPU, and the encoder's motion vectors are essentially free --
  they can drive the motion gate instead of frame differencing, saving more CPU.

## The trade, stated plainly

The Pi 5 + HAT is the better inference machine and the worse video machine. The
Pi 4B is the reverse. Which matters more depends on whether the bottleneck turns
out to be classification or continuous encoding -- and with sampled-frame
classification plus track voting, continuous encoding is the larger constant
load. Measure both on real hardware before committing.

## MACs are a weak proxy on a CPU, a good one on an NPU

If the Pi 4B is chosen, revisit the backbone. MAC count predicts NPU cost well
and ARM CPU cost poorly: squeeze-and-excite blocks stall the pipeline with a
full-tensor reduction, and swish is transcendental where ReLU6 is a clamp.
`efficientnet_lite0` exists for exactly this case -- EfficientNet-B0 with the SE
blocks removed and swish replaced, for mobile CPU INT8. Similar MACs, different
CPU behaviour.

## Quantisation

`birdcam.export.quantize` performs INT8 post-training quantisation and reports
per-class accuracy delta. Read it before shipping either way: fine-grained
classes sit close together in feature space and INT8 can smear them, rarely
evenly across classes.

The calibration set MUST be real feeder crops once they exist. Web photographs
have different noise, blur and exposure statistics than a camera at a feeder.
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="Export the two-head model to ONNX.")
    ap.add_argument("--role", default="student", choices=["student", "teacher", "local"])
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--from-frozen-head", default=None,
                    help="feature .npy stem to train real heads from")
    args = ap.parse_args()

    cfg = load_config()
    if args.from_frozen_head:
        path = export_from_frozen_head(cfg, args.from_frozen_head)
    else:
        path = export(cfg, role=args.role, checkpoint=args.checkpoint)
    note = cfg.path("reports_dir") / "deployment.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(DEPLOY_NOTE, encoding="utf-8")
    print(f"\nONNX:     {path}")
    print(f"metadata: {path.with_suffix('.json')}")
    print(f"deploy note: {note}")


if __name__ == "__main__":
    main()
