"""Assembling the service from one config file.

Every wiring decision lives here, so `__main__` stays a thin argument parser
and the components stay ignorant of configuration. Each builder resolves paths
against the repo root and fails with a message naming the config key rather
than a traceback from three layers down.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from capture.classifier import (
    BirdcamClipClassifier,
    ClassifierUnavailable,
    OnnxBackbone,
    build_novelty_scorer,
    check_artefact_pairing,
    load_operating_points,
    load_range_prior,
)
from capture.config import CaptureConfig
from capture.motion import MockMotionSource, PirMotionSource
from capture.pipeline import CapturePipeline
from capture.publisher import build_local_publisher
from capture.recorder import Picamera2Recorder, ReplayRecorder
from capture.service import CaptureService, TriggerGate
from capture.spool import Spool

logger = logging.getLogger(__name__)


def ensure_importable(root: Path) -> None:
    """Make `birdcam` and `web` importable when running from a checkout.

    Both are normally installed into the venv, but the service must also run
    straight from a clone -- which is how it gets tested before anything is
    installed on the Pi.
    """
    for candidate in (root, root / "ml" / "src"):
        path = str(candidate)
        if candidate.is_dir() and path not in sys.path:
            sys.path.insert(0, path)


def load_birdcam_config(root: Path) -> Any | None:
    """The ML project's Config, for the label space and common names.

    Returns None rather than raising: a missing or invalid ML config disables
    classification and degrades label rendering, but it should not stop a
    footage-collection run.
    """
    ensure_importable(root)
    try:
        from birdcam.config import load_config

        return load_config(root / "ml")
    except Exception as exc:  # noqa: BLE001 - any failure here is non-fatal
        logger.warning(
            "could not load the birdcam config (%s: %s); species common names "
            "will fall back to slugs",
            type(exc).__name__,
            exc,
        )
        return None


def build_spool(cfg: CaptureConfig) -> Spool:
    return Spool(
        work_dir=cfg.resolve_path("storage.work_dir"),
        pending_dir=cfg.resolve_path("storage.pending_dir"),
        review_dir=cfg.resolve_path("storage.review_dir"),
        min_free_mb=int(cfg.get("storage.min_free_mb")),
        max_pending_clips=int(cfg.get("storage.max_pending_clips")),
        max_review_clips=int(cfg.get("storage.max_review_clips")),
    )


def build_recorder(cfg: CaptureConfig, replay: Path | None = None):
    if replay is not None:
        logger.info("recorder: replaying %s instead of using the camera", replay)
        return ReplayRecorder(replay)
    cam = cfg.section("camera")
    return Picamera2Recorder(
        width=int(cam["width"]),
        height=int(cam["height"]),
        framerate=int(cam["framerate"]),
        bitrate_kbps=int(cam["bitrate_kbps"]),
        warmup_seconds=float(cam["warmup_seconds"]),
    )


def build_motion_source(cfg: CaptureConfig, *, mock: bool = False, schedule=None):
    if mock:
        return MockMotionSource(schedule=schedule)
    gpio = cfg.section("gpio")
    return PirMotionSource(
        pin=int(gpio["pin"]),
        sample_rate_hz=float(gpio["sample_rate_hz"]),
        queue_len=int(gpio["queue_len"]),
        warmup_seconds=float(gpio["warmup_seconds"]),
    )


def build_gate(cfg: CaptureConfig) -> TriggerGate:
    cap = cfg.section("capture")
    return TriggerGate(
        cooldown_seconds=float(cap["cooldown_seconds"]),
        on_busy=str(cap["on_busy"]),
        max_queued=int(cap["max_queued"]),
    )


def build_clip_classifier(cfg: CaptureConfig, birdcam_config: Any | None):
    """The real ONNX-backed classifier, or None when disabled.

    Raises ClassifierUnavailable when it is enabled but cannot be assembled --
    a service that silently runs without a classifier would delete nothing and
    publish everything, or publish nothing at all, depending on the policy. A
    misconfigured model must be a startup failure, not a runtime surprise.
    """
    section = cfg.section("classifier")
    if not section.get("enabled"):
        logger.warning(
            "CLASSIFICATION DISABLED (classifier.enabled: false). Clips will be "
            "recorded and retained unclassified; nothing is published."
        )
        return None

    if birdcam_config is None:
        raise ClassifierUnavailable(
            "classification is enabled but the birdcam config could not be "
            "loaded. The label space, thresholds and capture allowlist all come "
            "from ml/config/; without them there is nothing to decide with."
        )

    onnx_path = cfg.resolve_path("classifier.onnx_path")
    sidecar_path = onnx_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise ClassifierUnavailable(
            f"no metadata sidecar at {sidecar_path}. Class order is meaningless "
            "without the label list, and a mismatch between graph and labels is "
            "silent -- export/to_onnx.py ships them together for this reason."
        )

    import json

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    # The same check `eval/extract.load_extraction` makes: a graph whose class
    # order differs from the current taxonomy produces confident wrong labels.
    graph_classes = [str(c) for c in sidecar.get("taxon_classes", [])]
    if graph_classes != list(birdcam_config.taxon_classes):
        raise ClassifierUnavailable(
            f"{sidecar_path.name} was exported with a different taxon class order "
            f"({len(graph_classes)} classes vs {len(birdcam_config.taxon_classes)} "
            "in ml/config/ now). Re-export before deploying."
        )

    operating_points = load_operating_points(cfg.resolve_path("classifier.operating_points"))
    temperature = float(operating_points.get("temperature", 1.0))

    problems = check_artefact_pairing(
        sidecar,
        operating_points,
        expect_source=section.get("expect_source") or None,
    )
    if problems:
        message = (
            "the exported graph and its calibration do not describe one model:\n  - "
            + "\n  - ".join(problems)
        )
        if not section.get("allow_artefact_mismatch"):
            raise ClassifierUnavailable(
                message + "\n\nRefusing to start. Both artefacts are individually valid, "
                "which is what makes the pairing dangerous: nothing would fail, "
                "the labels would simply be wrong. Set "
                "classifier.allow_artefact_mismatch: true to override "
                "deliberately."
            )
        logger.error("ARTEFACT MISMATCH OVERRIDDEN BY CONFIG -- %s", message)

    site_prior = section.get("site_prior")
    range_prior = load_range_prior(
        cfg.resolve_path("classifier.site_prior") if site_prior else None
    )

    backbone = OnnxBackbone(
        onnx_path=onnx_path,
        providers=list(section["providers"]),
        image_size=int(sidecar.get("image_size", 224)),
    )
    novelty = build_novelty_scorer(section["novelty"], graph_emits_features=backbone.emits_features)

    from birdcam.inference import Classifier

    decider = Classifier(
        birdcam_config,
        novelty_scorer=novelty,
        temperature=temperature,
        range_prior=range_prior,
    )
    logger.info(
        "classifier ready: T=%.3f, %d range-prior weights, %d capture targets",
        temperature,
        len(range_prior),
        len(decider._capture_targets),
    )

    return BirdcamClipClassifier(
        backbone,
        decider,
        sample_fps=float(section["sample_fps"]),
        max_frames=int(section["max_frames"]),
        keep_frames=bool(section.get("keep_frames")),
        empty_gate=_build_empty_gate(cfg) if _gate_placement(cfg) == "after_record" else None,
    )


def build_pipeline(cfg: CaptureConfig, *, spool, recorder, classifier, publisher):
    pub = cfg.section("publish")
    return CapturePipeline(
        spool=spool,
        recorder=recorder,
        classifier=classifier,
        publisher=publisher,
        clip_seconds=float(cfg.get("capture.clip_seconds")),
        retain_uncertain=bool(pub["retain_uncertain"]),
        delete_after_publish=bool(cfg.get("storage.delete_after_publish")),
        escalate_after_attempts=int(pub["escalate_after_attempts"]),
        backoff_initial_seconds=float(pub["backoff_initial_seconds"]),
        backoff_max_seconds=float(pub["backoff_max_seconds"]),
        backoff_factor=float(pub["backoff_factor"]),
        empty_gate=_build_empty_gate(cfg) if _gate_placement(cfg) == "before_record" else None,
    )


def _gate_placement(cfg: CaptureConfig) -> str:
    """Where the empty gate runs: before the recorder starts, or on the clip.

    `after_record` is the default, and the reason is a measurement on the target.
    Running the gate first needs frames before the recorder starts, and on a Pi
    4B with Camera Module 3 a `peek()` took 1.76-2.03 s -- the camera is stopped
    after every recording and has to restart and settle. So every recording
    began ~2 s after the PIR fired, which for a sunbird that sips for two or
    three seconds loses the arrival or the whole visit.

    What the gate saves on this board is CLASSIFICATION, not the encode: the Pi
    4B keeps its hardware H.264 encoder, so recording is nearly free. The
    classifier is skipped just as well after recording, with no delay. On a
    Pi 5 -- no hardware encoder -- `before_record` would be the better trade.
    """
    from capture.config import CaptureConfigError

    try:
        placement = str(cfg.get("empty_gate.placement"))
    except CaptureConfigError:
        placement = "after_record"
    if placement not in ("after_record", "before_record"):
        raise CaptureConfigError(
            f"empty_gate.placement must be after_record or before_record, got {placement!r}"
        )
    return placement


def _build_empty_gate(cfg: CaptureConfig):
    """The geometric empty-feeder gate, unless it is switched off.

    Returns None when disabled, which makes the pipeline skip the check -- the
    fail-open direction. The classifier cannot answer this question: measured
    2026-09-11, its `empty_feeder` class scored recall 1.000 on backgrounds it
    trained on and 0.000 on one it had not seen.
    """
    from capture import emptygate as eg
    from capture.config import CaptureConfigError

    def opt(key: str, default):
        # cfg.get is deliberately strict -- it names a missing key rather than
        # guessing. This section is optional so that a config written before the
        # gate existed still starts, so absence is handled here instead of by
        # weakening that guarantee for every other key.
        try:
            return cfg.get(f"empty_gate.{key}")
        except CaptureConfigError:
            return default

    if not bool(opt("enabled", True)):
        logger.info("empty-feeder gate disabled by config")
        return None
    max_age = opt("max_age_seconds", eg.DEFAULT_MAX_AGE_SECONDS)
    try:
        gate = eg.IdleBackgroundGate(
            threshold=float(opt("threshold", eg.DEFAULT_THRESHOLD)),
            background_frames=int(opt("background_frames", eg.DEFAULT_IDLE_FRAMES)),
            snapshot_interval_seconds=float(
                opt("snapshot_interval_seconds", eg.DEFAULT_SNAPSHOT_INTERVAL_SECONDS)
            ),
            quiet_seconds=float(opt("quiet_seconds", eg.DEFAULT_QUIET_SECONDS)),
            max_age_seconds=None if max_age is None else float(max_age),
        )
        # An optional known-empty seed, for replay where there are no quiet
        # periods to snapshot. In the field, real snapshots displace it.
        if opt("reference", None):
            seed = eg.load_frame(cfg.resolve_path("empty_gate.reference"))
            if seed is None:
                raise CaptureConfigError("cannot read empty_gate.reference")
            gate.seed(seed)
    except ValueError as exc:
        raise CaptureConfigError(f"invalid empty_gate settings: {exc}") from exc
    return gate


def build_service(
    cfg: CaptureConfig,
    *,
    mock: bool = False,
    replay: Path | None = None,
    schedule=None,
) -> tuple[CaptureService, CapturePipeline]:
    """Everything, wired. Returns the service and the pipeline it drives."""
    ensure_importable(cfg.root)
    birdcam_config = load_birdcam_config(cfg.root)

    spool = build_spool(cfg)
    recorder = build_recorder(cfg, replay=replay)
    classifier = build_clip_classifier(cfg, birdcam_config)
    publisher = build_local_publisher(birdcam_config)
    pipeline = build_pipeline(
        cfg, spool=spool, recorder=recorder, classifier=classifier, publisher=publisher
    )
    gate = build_gate(cfg)
    motion = build_motion_source(cfg, mock=mock, schedule=schedule)

    service = CaptureService(
        motion_source=motion,
        pipeline=pipeline,
        gate=gate,
        recorder=recorder,
        empty_gate=pipeline.empty_gate or getattr(classifier, "empty_gate", None),
        drain_interval_seconds=max(5.0, float(cfg.get("publish.backoff_initial_seconds"))),
    )
    return service, pipeline
