"""INT8 post-training quantisation, with per-class accuracy delta.

Quantisation is the one compute saving that does not cost resolution. Reducing
the input size was measured and rejected -- 224 -> 160px halves the MACs and
costs 8.4 accuracy points, because fine-grained species ID depends on exactly
the detail that downsampling discards. INT8 keeps the resolution and shrinks the
arithmetic instead.

## Why per-class, and why loudly

Fine-grained classification is unusually sensitive to INT8. Nearly-identical
classes sit close together in feature space, and quantisation smears them --
but not evenly. An averaged accuracy delta can look harmless while a single
hard class collapses. This module reports the delta per class and flags any
that degrade beyond a configured threshold.

## Calibration data

The calibration set determines the activation ranges, and it must resemble what
the model will actually see. Web photographs have different noise, blur and
exposure statistics than a camera at a feeder, so the current set is marked
PROVISIONAL in the output. Replace it with real feeder crops as soon as they
exist -- this is the single highest-value substitution in the export path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from birdcam.config import Config, load_config

logger = logging.getLogger(__name__)


class _CalibrationReader:
    """Feeds representative inputs to the quantiser.

    Implements onnxruntime's CalibrationDataReader interface without importing
    it at module scope, so the base install stays light.
    """

    def __init__(self, samples: np.ndarray, input_name: str) -> None:
        self.samples = samples
        self.input_name = input_name
        self._i = 0

    def get_next(self):
        if self._i >= len(self.samples):
            return None
        item = {self.input_name: self.samples[self._i : self._i + 1]}
        self._i += 1
        return item

    def rewind(self):
        self._i = 0


def build_calibration_set(cfg: Config, n: int, image_size: int, seed: int = 0) -> np.ndarray:
    """Sample preprocessed images from the TRAIN split as calibration data.

    Train, never test: calibration is part of fitting the model, and using test
    images would leak.
    """
    import timm
    from PIL import Image

    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest

    with open_manifest(cfg.path("manifest_db")) as m:
        items = [i for i in load_labelled(cfg, m) if i.split == "train"]
    rng = np.random.RandomState(seed)
    picked = [items[i] for i in rng.choice(len(items), min(n, len(items)), replace=False)]

    model = timm.create_model(
        cfg.train_cfg["backbone"]["student"]["name"], pretrained=False, num_classes=0
    )
    dc = timm.data.resolve_data_config({}, model=model)
    dc["input_size"] = (3, image_size, image_size)
    tf = timm.data.create_transform(**dc, is_training=False)

    out = np.zeros((len(picked), 3, image_size, image_size), dtype=np.float32)
    for k, it in enumerate(picked):
        with Image.open(it.path) as im:
            out[k] = tf(im.convert("RGB")).numpy()
    logger.info("calibration set: %d images from the train split", len(out))
    return out


def quantize(cfg: Config, fp32_path: Path | None = None, out_path: Path | None = None) -> Path:
    """Static INT8 quantisation of the exported ONNX graph."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    qc = cfg.train_cfg["export"]["quantize"]
    fp32_path = fp32_path or (cfg.root / cfg.train_cfg["export"]["onnx_path"])
    if not fp32_path.is_file():
        raise RuntimeError(f"no FP32 model at {fp32_path}; run birdcam.export.to_onnx first")
    out_path = out_path or fp32_path.with_name(fp32_path.stem + "_int8.onnx")

    size = cfg.train_cfg["backbone"]["student"]["image_size"]
    calib = build_calibration_set(cfg, qc["calibration_size"], size)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Shape inference first; quantize_static is unreliable without it.
    #
    # skip_symbolic_shape=True is required: the exported graph has a dynamic
    # batch axis, and ORT's symbolic shape inference raises "Incomplete symbolic
    # shape inference" on it. Static shape inference handles the graph fine
    # because only the batch dimension is dynamic -- every spatial dimension is
    # fixed, which is what quantisation actually needs to know.
    prepped = fp32_path.with_name(fp32_path.stem + "_prep.onnx")
    quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=True)

    quantize_static(
        str(prepped),
        str(out_path),
        _CalibrationReader(calib, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,  # per-channel weights matter for depthwise convs
    )
    prepped.unlink(missing_ok=True)
    logger.info(
        "INT8 model: %s (%.1f MB, was %.1f MB)",
        out_path, out_path.stat().st_size / 1e6, fp32_path.stat().st_size / 1e6,
    )
    return out_path


def compare(cfg: Config, fp32_path: Path, int8_path: Path, limit: int | None = None) -> dict:
    """Run both models over the test split and report per-class deltas."""
    import onnxruntime as ort
    import timm
    from PIL import Image

    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest

    with open_manifest(cfg.path("manifest_db")) as m:
        items = [i for i in load_labelled(cfg, m) if i.split == "test"]
    if limit:
        items = items[:limit]

    size = cfg.train_cfg["backbone"]["student"]["image_size"]
    model = timm.create_model(
        cfg.train_cfg["backbone"]["student"]["name"], pretrained=False, num_classes=0
    )
    dc = timm.data.resolve_data_config({}, model=model)
    dc["input_size"] = (3, size, size)
    tf = timm.data.create_transform(**dc, is_training=False)

    s32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    name32 = s32.get_inputs()[0].name
    name8 = s8.get_inputs()[0].name

    y, p32, p8 = [], [], []
    for k, it in enumerate(items):
        with Image.open(it.path) as im:
            x = tf(im.convert("RGB")).numpy()[None]
        p32.append(int(s32.run(None, {name32: x})[0].argmax()))
        p8.append(int(s8.run(None, {name8: x})[0].argmax()))
        y.append(it.taxon_index)
        if (k + 1) % 200 == 0:
            logger.info("  %d/%d", k + 1, len(items))

    y, p32, p8 = np.array(y), np.array(p32), np.array(p8)
    flag = cfg.train_cfg["export"]["quantize"]["flag_threshold_pct"] / 100.0

    per_class = []
    for c in sorted(set(y.tolist())):
        sel = y == c
        a32 = float((p32[sel] == c).mean())
        a8 = float((p8[sel] == c).mean())
        per_class.append(
            {
                "class": cfg.taxon_classes[c],
                "n": int(sel.sum()),
                "fp32": round(a32, 4),
                "int8": round(a8, 4),
                "delta": round(a8 - a32, 4),
                "flagged": (a32 - a8) > flag,
            }
        )
    per_class.sort(key=lambda r: r["delta"])

    return {
        "n_test": len(items),
        "fp32_accuracy": round(float((p32 == y).mean()), 4),
        "int8_accuracy": round(float((p8 == y).mean()), 4),
        "agreement": round(float((p32 == p8).mean()), 4),
        "fp32_mb": round(fp32_path.stat().st_size / 1e6, 1),
        "int8_mb": round(int8_path.stat().st_size / 1e6, 1),
        "calibration_source": cfg.train_cfg["export"]["quantize"]["calibration_source"],
        "per_class": per_class,
    }


def print_report(res: dict) -> None:
    print(f"\nFP32 {res['fp32_mb']} MB -> INT8 {res['int8_mb']} MB "
          f"({res['fp32_mb'] / max(res['int8_mb'], 1e-9):.1f}x smaller)")
    print(f"accuracy {res['fp32_accuracy']:.4f} -> {res['int8_accuracy']:.4f} "
          f"({res['int8_accuracy'] - res['fp32_accuracy']:+.4f})")
    print(f"the two models agree on {res['agreement']:.1%} of test images")

    if res["calibration_source"].startswith("provisional"):
        print("\n*** CALIBRATION SET IS PROVISIONAL ***")
        print("Ranges were fitted on web photographs, not feeder crops. Real")
        print("captures have different noise, blur and exposure statistics.")
        print("Re-run this once real crops exist -- these numbers will move.")

    flagged = [r for r in res["per_class"] if r["flagged"]]
    print(f"\n{'class':<30}{'n':>5}{'fp32':>8}{'int8':>8}{'delta':>8}")
    print("-" * 59)
    for r in res["per_class"][:10]:
        mark = "  <-- FLAGGED" if r["flagged"] else ""
        print(f"{r['class'].replace('_',' '):<30}{r['n']:>5}{r['fp32']:>8.3f}"
              f"{r['int8']:>8.3f}{r['delta']:>+8.3f}{mark}")
    if flagged:
        print(f"\n{len(flagged)} class(es) degraded beyond threshold. Fine-grained classes")
        print("sit close together in feature space and INT8 smears them unevenly --")
        print("an averaged delta would have hidden this.")
    else:
        print("\nNo class degraded beyond the configured threshold.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="INT8 quantise and report per-class delta.")
    ap.add_argument("--limit", type=int, default=None, help="test images to evaluate")
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    fp32 = cfg.root / cfg.train_cfg["export"]["onnx_path"]
    int8 = quantize(cfg)
    if args.skip_compare:
        return
    res = compare(cfg, fp32, int8, limit=args.limit)
    print_report(res)
    out = cfg.path("reports_dir") / "quantisation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
