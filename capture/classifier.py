"""Clip -> one Decision, using the existing inference algorithm unchanged.

The integration contract, found in `ml/src/birdcam/inference.py`, is
`Classifier.decide(taxon_logits, sex_logits, features) -> Decision` plus
`Classifier.vote(decisions) -> Decision`. It takes model outputs, not files.
Everything between "an mp4 exists" and "logits exist" is this module's job, and
nothing here re-implements any part of the decision itself.

Frames, not video
-----------------
`reports/deployment.md` is explicit for the Pi 4B: "Classify SAMPLED frames
only, never every frame. A visit lasting seconds yields plenty; the track vote
does the rest." So a clip is decoded at a low frame rate and each sampled frame
gets its own `decide()`, then `vote()` combines them -- which is also the path
`vote()` was written for (unknown frames are counted but do not outvote
confident bird frames).

Preprocessing lives here, on purpose
------------------------------------
`export/to_onnx.py` states the exported graph is "preprocessed tensor in, two
logit tensors out" and that "normalisation stays in the capture application".
So `preprocess_frame` reproduces `build_eval_transform` -- resize the short side
to 1.14x, centre-crop, scale to [0,1], ImageNet mean/std -- in numpy and PIL,
with no torch on the Pi.

The A26 guard
-------------
ASSUMPTIONS.md A26 records that the exported ONNX and the configured thresholds
currently describe **different models**: a frozen-feature export paired with
thresholds fitted on the fine-tuned checkpoint. Both artefacts are individually
valid, which is what makes the pairing dangerous -- nothing fails, the numbers
are simply wrong. `check_artefact_pairing` refuses to start on that combination
rather than producing plausible wrong labels for weeks.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet statistics, matching birdcam.train.augment.build_eval_transform.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# torchvision's Resize is given int(image_size * 1.14) before the centre crop.
_RESIZE_RATIO = 1.14
# Short side the frames are decoded at, matching data/field.py's extraction so
# a clip scored here goes through the same pixels as one scored offline.
_EXTRACT_SHORT_SIDE = 256


class ClassifierUnavailable(RuntimeError):
    """The model or its calibration artefacts could not be loaded."""


@dataclass
class ClipResult:
    """What one clip produced."""

    decision: Any | None  # birdcam.inference.Decision
    frames_scored: int
    keyframe: Path | None = None
    per_frame: list[Any] | None = None


class ClipClassifier(Protocol):
    def classify(self, clip: Path, work_dir: Path, event_id: str) -> ClipResult: ...


# -- frame sampling ------------------------------------------------------------


def sample_frames(
    clip: Path,
    out_dir: Path,
    *,
    fps: float,
    max_frames: int,
    short_side: int = _EXTRACT_SHORT_SIDE,
) -> list[Path]:
    """Decode `clip` to JPEGs at `fps`, capped at `max_frames`.

    Zero-byte frames at a clip boundary are a real occurrence -- data/field.py
    hit one in 11,051 -- and one bad frame must cost one frame, not the event.
    """
    if shutil.which("ffmpeg") is None:
        raise ClassifierUnavailable(
            "ffmpeg is not on PATH; frame sampling needs it.\n  sudo apt install -y ffmpeg"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%03d.jpg")
    cmd = [
        "ffmpeg", "-v", "error", "-threads", "2", "-i", str(clip),
        "-vf", f"fps={fps},scale=-1:{short_side}",
        "-frames:v", str(int(max_frames)),
        "-q:v", "3", "-y", pattern,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ClassifierUnavailable(f"ffmpeg failed on {clip.name}: {proc.stderr[-300:]}")

    frames = sorted(out_dir.glob("frame_[0-9][0-9][0-9].jpg"))
    good = [p for p in frames if p.stat().st_size > 0]
    if len(good) != len(frames):
        logger.warning("dropped %d zero-byte frame(s) from %s", len(frames) - len(good), clip.name)
    return good


def preprocess_frame(path: Path, image_size: int) -> np.ndarray:
    """One JPEG -> (3, H, W) float32, matching build_eval_transform."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        target = int(image_size * _RESIZE_RATIO)
        w, h = im.size
        # A single int to torchvision's Resize scales the SHORT side.
        if w < h:
            new_w, new_h = target, max(1, round(h * target / w))
        else:
            new_h, new_w = target, max(1, round(w * target / h))
        im = im.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - image_size) // 2
        top = (new_h - image_size) // 2
        im = im.crop((left, top, left + image_size, top + image_size))
        arr = np.asarray(im, dtype=np.float32) / 255.0

    arr = (arr - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


# -- the ONNX session ----------------------------------------------------------


class OnnxBackbone:
    """Thin wrapper over an onnxruntime session for the two-head graph."""

    def __init__(self, onnx_path: Path, providers: list[str], image_size: int) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ClassifierUnavailable(
                "onnxruntime is not installed.\n  pip install onnxruntime\n"
                "On a Pi 4B prefer the XNNPACK execution provider: it has NEON "
                "INT8 kernels (see reports/deployment.md)."
            ) from exc

        if not Path(onnx_path).is_file():
            raise ClassifierUnavailable(f"no ONNX model at {onnx_path}")

        available = set(ort.get_available_providers())
        wanted = [p for p in providers if p in available]
        if not wanted:
            raise ClassifierUnavailable(
                f"none of the configured providers {providers} are available. "
                f"onnxruntime offers: {sorted(available)}"
            )
        if wanted != list(providers):
            logger.warning(
                "requested providers %s, using %s (the rest are not built into this onnxruntime)",
                providers,
                wanted,
            )

        self.session = ort.InferenceSession(str(onnx_path), providers=wanted)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.image_size = int(image_size)
        logger.info(
            "ONNX session ready: %s, providers=%s, outputs=%s",
            Path(onnx_path).name,
            wanted,
            self.output_names,
        )

    @property
    def emits_features(self) -> bool:
        """Whether the graph exposes pooled features.

        The current export does not (A26: "the ONNX graph emits only
        taxon_logits and sex_logits"), which is what confines the novelty gate
        to logits-only scorers.
        """
        return "features" in self.output_names

    def run(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: batch})
        return dict(zip(self.output_names, outputs, strict=False))


# -- the clip classifier -------------------------------------------------------


class BirdcamClipClassifier:
    """Samples frames, runs the graph, and defers every judgement to birdcam."""

    def __init__(
        self,
        backbone: OnnxBackbone,
        decider: Any,  # birdcam.inference.Classifier
        *,
        sample_fps: float,
        max_frames: int,
        keep_frames: bool = False,
    ) -> None:
        self.backbone = backbone
        self.decider = decider
        self.sample_fps = float(sample_fps)
        self.max_frames = int(max_frames)
        self.keep_frames = keep_frames

    def classify(self, clip: Path, work_dir: Path, event_id: str) -> ClipResult:
        frame_dir = work_dir / f"{event_id}_frames"
        try:
            frames = sample_frames(clip, frame_dir, fps=self.sample_fps, max_frames=self.max_frames)
            if not frames:
                logger.warning("%s yielded no usable frames", clip.name)
                return ClipResult(decision=None, frames_scored=0)

            batch = np.stack([preprocess_frame(p, self.backbone.image_size) for p in frames])
            out = self.backbone.run(batch)
            taxon = out["taxon_logits"]
            sex = out.get("sex_logits")
            features = out.get("features")

            decisions = []
            for i in range(len(frames)):
                decisions.append(
                    self.decider.decide(
                        taxon[i],
                        sex[i] if sex is not None else None,
                        features[i] if features is not None else None,
                    )
                )

            voted = self.decider.vote(decisions)
            keyframe = self._pick_keyframe(frames, decisions, voted, work_dir, event_id)
            return ClipResult(
                decision=voted,
                frames_scored=len(frames),
                keyframe=keyframe,
                per_frame=decisions,
            )
        finally:
            if not self.keep_frames:
                shutil.rmtree(frame_dir, ignore_errors=True)

    def _pick_keyframe(
        self,
        frames: list[Path],
        decisions: list[Any],
        voted: Any,
        work_dir: Path,
        event_id: str,
    ) -> Path | None:
        """The frame that best represents the voted label.

        Preferring a frame that agrees with the vote matters: the thumbnail is
        what a human checks the label against, and a thumbnail showing a
        different bird from the one in the row is worse than no thumbnail.
        """
        if not frames:
            return None
        agreeing = [
            (d.confidence, p)
            for p, d in zip(frames, decisions, strict=False)
            if getattr(d, "label", None) == getattr(voted, "label", None)
        ]
        pool = agreeing or [
            (getattr(d, "confidence", 0.0), p) for p, d in zip(frames, decisions, strict=False)
        ]
        best = max(pool, key=lambda t: t[0])[1]
        dest = work_dir / f"{event_id}.jpg"
        shutil.copyfile(best, dest)
        return dest


class StubClipClassifier:
    """Returns a fixed Decision. For tests and for exercising the plumbing.

    Lets the full state machine, keep/discard rule, spool and publisher run with
    no model present at all -- which is how the decision logic stays testable on
    a machine that has neither the ONNX export nor a camera.
    """

    def __init__(self, decision_factory) -> None:
        self.decision_factory = decision_factory
        self.calls = 0

    def classify(self, clip: Path, work_dir: Path, event_id: str) -> ClipResult:
        self.calls += 1
        return ClipResult(decision=self.decision_factory(), frames_scored=1, keyframe=None)


# -- artefact loading and the A26 guard ----------------------------------------


_SHA_TOKEN = re.compile(r"\b([0-9a-f]{12,64})\b")


def _sha_in(text: str) -> str:
    """The checkpoint SHA embedded in a `source` string, or ''.

    `write_thresholds_to_config` formats it as
    "fine-tuned checkpoint student_best.pt @ a71e95cca471".
    """
    match = _SHA_TOKEN.search(text.lower())
    return match.group(1) if match else ""


def check_artefact_pairing(
    sidecar: dict[str, Any],
    operating_points: dict[str, Any],
    *,
    expect_source: str | None = None,
) -> list[str]:
    """Return reasons the graph and the calibration do not describe one model.

    Empty list means the pairing is coherent as far as the metadata can show.
    """
    problems: list[str] = []
    training = str(sidecar.get("training", "")).lower()
    source = str(operating_points.get("source", ""))

    # Positive check first, where the metadata supports one. Both sides now
    # stamp the checkpoint SHA (to_onnx writes `checkpoint_sha`,
    # write_thresholds_to_config writes it into `source`), so the pairing can
    # be CONFIRMED rather than merely not-contradicted. Without this the
    # string heuristics below pass whenever a sidecar simply says nothing,
    # which is the state that produced A26 in the first place.
    sidecar_sha = str(sidecar.get("checkpoint_sha") or "")
    source_sha = _sha_in(source)
    if sidecar_sha and source_sha:
        if not sidecar_sha.startswith(source_sha):
            problems.append(
                f"the ONNX graph was exported from checkpoint {sidecar_sha[:12]} "
                f"but the calibration was fitted on {source_sha}. Different "
                "checkpoints: the thresholds and temperature do not belong to "
                "this graph"
            )
    elif not sidecar_sha:
        logger.warning(
            "the ONNX sidecar carries no checkpoint_sha, so the pairing with %r "
            "cannot be positively verified -- only obvious contradictions are "
            "caught. Re-export with a current birdcam.export.to_onnx to stamp it "
            "(ASSUMPTIONS.md A26).",
            source or "the operating points",
        )

    if not sidecar.get("trained", False):
        problems.append(
            "the ONNX sidecar says trained=false -- this graph holds "
            "randomly-initialised weights and its predictions are noise"
        )

    if "frozen-feature" in training and "fine-tuned" in source.lower():
        problems.append(
            f"the ONNX graph is a frozen-feature export ({sidecar.get('training')!r}) "
            f"but the calibration was fitted on {source!r}. These are different "
            "models; pairing them applies fine-tuned thresholds and temperature "
            "to a frozen-feature graph. See ASSUMPTIONS.md A26"
        )

    if expect_source and expect_source not in source:
        problems.append(
            f"classifier.expect_source is {expect_source!r} but the operating "
            f"points were fitted on {source!r}"
        )

    return problems


def load_operating_points(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ClassifierUnavailable(
            f"no operating points at {path}. Temperature and the per-class "
            "thresholds are fitted by `birdcam.eval.thresholds`; without them "
            "the confidence numbers this service records are uncalibrated, and "
            "confidence gates the capture decision."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_range_prior(path: Path | None) -> dict[str, float]:
    """Per-class weights from a site config, e.g. config/sites/rondebosch.yaml."""
    if path is None:
        return {}
    import yaml

    if not Path(path).is_file():
        raise ClassifierUnavailable(f"no site prior at {path}")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    weights = data.get("weights") or {}
    if not weights:
        logger.warning(
            "%s has no populated `weights` block; running with no range prior. "
            "Nothing will tell the model that a Cape Town feeder sees six "
            "species routinely (ASSUMPTIONS.md A21).",
            path,
        )
    return {str(k): float(v) for k, v in weights.items()}


def build_novelty_scorer(cfg: dict[str, Any], *, graph_emits_features: bool):
    """Construct the open-set failsafe, or return None with a loud warning.

    THE SEAM. A25/A26 record that the deployment bundle cannot currently
    reconstruct the kNN scorer: the ONNX graph emits logits only, and the kNN
    scorer needs the 1,280-d pooled features from the FROZEN backbone plus its
    reference vectors and threshold, none of which are serialised. Until the
    inference side ships those, this returns None and `Classifier.decide()`
    skips step 1 entirely.

    `scorer: energy` works today and needs nothing extra -- it scores the logits
    the graph already emits (reports/open_set.json: AUROC 0.927, 68.1% of OOD
    caught at 5% false-alarm rate). It is the weaker detector, but it is a real
    gate rather than no gate.
    """
    if not cfg.get("enabled", False):
        logger.warning(
            "OPEN-SET FAILSAFE DISABLED. Every frame will be forced into one of "
            "the taxon classes, so a squirrel is reported as the least-bad bird "
            "rather than `unknown`. Set classifier.novelty.enabled to turn this "
            "on; see ASSUMPTIONS.md A25/A26 for what still has to be serialised."
        )
        return None

    kind = str(cfg.get("scorer", "energy")).lower()
    threshold = cfg.get("threshold")
    if threshold is None:
        raise ClassifierUnavailable(
            "classifier.novelty.enabled is true but no threshold is set. The "
            "threshold is an operational choice -- pick one from "
            "reports/open_set.json at the false-alarm rate you want."
        )

    from birdcam.models.novelty import EnergyScorer, MaxSoftmaxScorer

    if kind == "energy":
        scorer = EnergyScorer(threshold=float(threshold))
    elif kind == "max_softmax":
        scorer = MaxSoftmaxScorer(threshold=float(threshold))
    elif kind == "knn":
        if not graph_emits_features:
            raise ClassifierUnavailable(
                "classifier.novelty.scorer is 'knn', which needs pooled backbone "
                "features, but the ONNX graph emits only taxon_logits and "
                "sex_logits (ASSUMPTIONS.md A26). Re-export with a features "
                "output, or use scorer: energy which needs logits only."
            )
        reference = cfg.get("reference")
        if not reference or not Path(reference).is_file():
            raise ClassifierUnavailable(
                f"classifier.novelty.reference is {reference!r}; the kNN scorer "
                "needs its reference feature vectors on disk."
            )
        # Loaded through the scorer's own API: the file layout, hyperparameters
        # and fitted state all belong to birdcam.models.novelty. Reaching in to
        # set `_ref` here (as this did) meant a change to what `fit()` stores
        # would break deployment silently.
        from birdcam.models.novelty import load_scorer

        scorer = load_scorer(reference)
        if scorer.name != "knn":
            raise ClassifierUnavailable(
                f"{reference} holds a {scorer.name!r} scorer but config asks for 'knn'"
            )
        bundled = scorer.threshold
        scorer.threshold = float(threshold)
        # Tolerance is operational, not numerical: a config author writes
        # 0.7009, the calibration produced 0.700861347343802, and warning about
        # that gap would train everyone to ignore the warning. Anything larger
        # is a deliberate override worth saying out loud.
        if bundled is not None and abs(bundled - float(threshold)) > 1e-3:
            logger.warning(
                "novelty threshold override: config says %.6f but the bundle was "
                "calibrated at %.6f. Using the config value. Drift between these "
                "is how a gate ends up carrying someone else's operating point.",
                float(threshold),
                bundled,
            )
    else:
        raise ClassifierUnavailable(f"unknown classifier.novelty.scorer {kind!r}")

    logger.info("open-set failsafe: %s scorer at threshold %.4f", kind, float(threshold))
    return scorer
