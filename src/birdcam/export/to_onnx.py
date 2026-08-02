"""PyTorch -> ONNX export for the two-head deployment model.

Kept working from early on, because ONNX is the pivot point for BOTH runtimes we
care about: ONNX Runtime (for CPU sanity checks and INT8 calibration) and the
Hailo Dataflow Compiler (which consumes ONNX to build a .hef for the Hailo-8L).
A break here blocks the entire deployment path, so it is exercised well before
there is a trained checkpoint to export.

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


def export(cfg: Config, role: str = "student", checkpoint: Path | None = None,
           out_path: Path | None = None, verify: bool = True) -> Path:
    """Export the two-head model to ONNX.

    Exports randomly-initialised weights when no checkpoint is supplied. That is
    useful for validating the graph and measuring its cost, but such a file must
    never be shipped -- the metadata sidecar records which case it was.
    """
    import torch

    from birdcam.models.heads import build_model

    model, backbone_name = build_model(cfg, role)
    trained = False
    if checkpoint and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        trained = True
        logger.info("loaded weights from %s", checkpoint)
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
        logger.warning("onnx/onnxruntime not installed; skipping verification "
                       "(uv sync --extra export)")
        return

    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, size, size).astype(np.float32)
    taxon, sex = sess.run(None, {"image": x})
    assert taxon.shape == (2, n_taxon), f"taxon shape {taxon.shape} != (2, {n_taxon})"
    assert sex.shape == (2, n_sex), f"sex shape {sex.shape} != (2, {n_sex})"
    logger.info("verified: taxon%s sex%s", taxon.shape, sex.shape)


HAILO_NOTE = """\
# Hailo-8L compilation (run separately, on x86 Linux)

The Hailo Dataflow Compiler is not a Python dependency of this repo and does not
run on the Pi. It is an x86-64 Linux toolchain, installed separately from the
Hailo Developer Zone (account required). Compilation happens on a workstation;
only the resulting .hef is copied to the Pi.

Outline (verify against the DFC version you install -- this has NOT been run):

    hailo parser onnx birdcam_student.onnx --hw-arch hailo8l
    hailo optimize birdcam_student.har --calib-set-path calib/ --hw-arch hailo8l
    hailo compiler birdcam_student_optimized.har --hw-arch hailo8l

Notes that matter:

* `hailo optimize` performs INT8 quantisation and needs a representative
  calibration set. That set MUST be real feeder crops once they exist -- web
  photographs have different noise, blur and exposure statistics than a Pi
  camera at a feeder, and calibrating on the wrong distribution is how
  fine-grained classes get smeared together.
* The student must remain a plain CNN. ViT and ConvNeXt blocks (LayerNorm,
  GELU, attention) either fail to compile or fall back to CPU for large
  subgraphs.
* Expect per-class accuracy loss from INT8 concentrated in the visually similar
  classes. Measure it per class, not as an average.
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="Export the two-head model to ONNX.")
    ap.add_argument("--role", default="student", choices=["student", "teacher", "local"])
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config()
    path = export(cfg, role=args.role, checkpoint=args.checkpoint)
    note = cfg.path("reports_dir") / "hailo_compilation.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(HAILO_NOTE, encoding="utf-8")
    print(f"\nONNX:     {path}")
    print(f"metadata: {path.with_suffix('.json')}")
    print(f"Hailo note: {note}")


if __name__ == "__main__":
    main()
