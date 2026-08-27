"""Entry point: `python -m capture`.

Modes
-----
    python -m capture                       run the service (PIR + camera)
    python -m capture --check               validate config and artefacts, exit
    python -m capture --classify CLIP       push one existing clip through the
                                            full decision pipeline, then exit
    python -m capture --mock --replay CLIP  run the service with a fake PIR and
                                            a replayed clip, no hardware at all

The last two are how the pipeline is exercised off-Pi. `--mock --replay` runs
the real state machine, the real keep/discard rule, the real spool and the real
publisher against the real dashboard database -- only the sensor and the camera
are substituted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from capture import build
from capture.config import CaptureConfig, CaptureConfigError
from capture.events import MotionEvent, Trigger
from capture.logging_setup import setup_logging

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "capture.yaml"

logger = logging.getLogger("capture")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="capture", description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"path to the capture config (default: {DEFAULT_CONFIG})",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="load the config, build every component, print a summary and exit",
    )
    ap.add_argument(
        "--classify",
        type=Path,
        metavar="CLIP",
        help="run one pre-recorded clip through classification and the "
        "keep/discard decision, then exit",
    )
    ap.add_argument(
        "--mock",
        action="store_true",
        help="use a mock PIR instead of GPIO (requires --replay)",
    )
    ap.add_argument(
        "--replay",
        type=Path,
        metavar="CLIP",
        help="use this clip instead of the camera",
    )
    ap.add_argument(
        "--triggers",
        type=int,
        default=1,
        help="with --mock, how many motion events to fire (default: 1)",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="with --mock, seconds between fired triggers (default: 2.0)",
    )
    ap.add_argument(
        "--log-level",
        default=None,
        help="override logging.level from the config",
    )
    return ap.parse_args(argv)


def _configure_logging(cfg: CaptureConfig, override: str | None) -> None:
    section = cfg.section("logging")
    log_file = cfg.resolve_path("logging.file") if section.get("file") else None
    setup_logging(
        level=override or section["level"],
        file=log_file,
        max_bytes=int(section["max_bytes"]),
        backup_count=int(section["backup_count"]),
        fmt=str(section["format"]),
    )


def _load(args: argparse.Namespace) -> CaptureConfig:
    if not args.config.is_file():
        raise SystemExit(
            f"No capture config at {args.config}.\n"
            "Copy the reference and edit it:\n"
            f"  cp capture/config/capture.example.yaml {args.config}"
        )
    return CaptureConfig.load(args.config)


def run_check(cfg: CaptureConfig) -> int:
    """Build everything without arming the sensor. The pre-flight check."""
    print(cfg.summary())
    print()

    build.ensure_importable(cfg.root)
    birdcam_config = build.load_birdcam_config(cfg.root)
    if birdcam_config is not None:
        print(f"taxon classes   : {len(birdcam_config.taxon_classes)}")
        print(f"tier A species  : {len(birdcam_config.species_by_tier('A'))}")

    spool = build.build_spool(cfg)
    print(f"free space      : {spool.free_mb():.0f} MB (floor {spool.min_free_mb} MB)")
    print(f"pending clips   : {spool.pending_count()} / {spool.max_pending_clips}")
    print(f"review clips    : {spool.review_count()} / {spool.max_review_clips}")

    try:
        classifier = build.build_clip_classifier(cfg, birdcam_config)
    except Exception as exc:  # noqa: BLE001 - --check exists to report exactly this
        print(f"\nCLASSIFIER: NOT USABLE\n{exc}")
        return 1

    if classifier is None:
        print("\nclassifier      : disabled")
    else:
        targets = sorted(classifier.decider._capture_targets)
        print(f"\ncapture allowlist ({len(targets)}):")
        for slug in targets:
            from capture.labels import display_name

            print(f"  {slug:<28} {display_name(slug, birdcam_config)}")

    print("\nconfig and artefacts are usable.")
    return 0


def run_one_clip(cfg: CaptureConfig, clip: Path) -> int:
    """Feed one existing clip through the pipeline, exactly as an event would."""
    if not clip.is_file():
        raise SystemExit(f"no clip at {clip}")

    build.ensure_importable(cfg.root)
    birdcam_config = build.load_birdcam_config(cfg.root)
    spool = build.build_spool(cfg)
    recorder = build.build_recorder(cfg, replay=clip)
    classifier = build.build_clip_classifier(cfg, birdcam_config)
    publisher = build.build_local_publisher(birdcam_config)
    pipeline = build.build_pipeline(
        cfg, spool=spool, recorder=recorder, classifier=classifier, publisher=publisher
    )

    event = MotionEvent.now(Trigger.MANUAL)
    logger.info("feeding %s through the pipeline as event %s", clip.name, event.event_id)
    record = pipeline.handle(event)
    pipeline.drain_pending()

    print()
    for key, value in record.log_fields().items():
        print(f"  {key:<18} {value}")
    return 0 if record.error is None else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = _load(args)
    except CaptureConfigError as exc:
        # Config errors are for a human to read, not a traceback to decode.
        print(str(exc), file=sys.stderr)
        return 2

    _configure_logging(cfg, args.log_level)

    if args.check:
        return run_check(cfg)

    if args.classify:
        return run_one_clip(cfg, args.classify)

    if args.mock and args.replay is None:
        raise SystemExit("--mock needs --replay CLIP: there is no camera to record from")

    schedule = [args.interval] * args.triggers if args.mock else None
    try:
        service, _pipeline = build.build_service(
            cfg, mock=args.mock, replay=args.replay, schedule=schedule
        )
    except Exception as exc:  # noqa: BLE001 - startup failures are operator-facing
        logger.error("cannot start: %s", exc)
        return 1

    service.install_signal_handlers()
    return service.run()


if __name__ == "__main__":
    sys.exit(main())
