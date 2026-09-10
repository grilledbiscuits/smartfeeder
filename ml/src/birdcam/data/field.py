"""Field footage: frame extraction and first-contact measurement.

This is the first real deployment-domain data the project has had. Everything
measured before it came from deliberate web photographs, and A10 has flagged
the web-to-feeder domain gap as the project's largest technical risk since
Phase 1. Nothing here trains anything; it exists to find out how far the gap
actually is.

What it does
------------
1. Decodes `ml/training/<species>/*.mp4` at a low frame rate. Sixty frames a
   second of the same perched bird is one sample, not sixty, so extracting
   every frame would only inflate the corpus with near-duplicates.
2. Records provenance for every frame: which clip, which recording SESSION,
   and the timestamp within the clip. Session is the unit that matters -- two
   cuts from the same session are the same bird in the same light, and putting
   them in different splits would leak.
3. Runs the fine-tuned checkpoint over the extracted frames and reports what
   it predicts.

Label semantics -- read this before trusting any number it prints
----------------------------------------------------------------
The folder name is a CLAIM about the primary bird in a clip, not a per-frame
label. Verified counterexample: a frame in `capewhiteeye/20260816_133907_1.mp4`
contains a Cape White-eye AND a sunbird on the chain above it. Frames may
contain several birds, or none, since a cut clip includes the moments before
and after a visit.

So the per-species figures below are NOT accuracy. They are "of frames drawn
from clips labelled X, here is what the model said", which is a measure of the
domain gap contaminated by an unknown amount of label noise. Treated as
accuracy they would be pessimistic in a way nobody could quantify.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from birdcam.names import display

logger = logging.getLogger(__name__)

PHASE = 8


@dataclass(frozen=True)
class FolderLabel:
    """What a footage folder claims, and whether that claim is a species.

    `resolved=False` marks a label that is deliberately coarser than a species
    because the species cannot be determined from the footage. Recording that
    distinction in the data matters more than it sounds: this project has twice
    been bitten by a confident label outliving its evidence (A20's superseded
    trigger rate, A26's mismatched export). A field that says "this is a group,
    not a species" cannot quietly become a species later.
    """

    label: str
    resolved: bool
    note: str = ""


# Folder name -> what that folder actually establishes. `uncut` is deliberately
# absent: it is unlabelled by nature and is the source of empty-feeder negatives.
FOLDER_TO_LABEL = {
    # 2026-09-10: identified as Southern Double-collared by the observer who
    # recorded it. This entry previously carried resolved=False because neither
    # a blind test on labelled corpus images (15/20, and 3/5 with no breast band
    # visible) nor the footage itself could separate Southern from Greater. That
    # is still true of the IMAGES; the resolution comes from the person who was
    # there, which is a better instrument than either of us had.
    "doublecollared": FolderLabel(
        "Cinnyris chalybeus",
        resolved=True,
        note="Field identification by the observer, 2026-09-10; not established from imagery.",
    ),
    "amethyst": FolderLabel("Chalcomitra amethystina", resolved=True),
    "juvenileamethyst": FolderLabel(
        "Chalcomitra amethystina",
        resolved=True,
        note="Female/immature plumage; one frame shows an immature male's amethyst gorget.",
    ),
    # `whiteeye` is the 2026-09 drive's name, `capewhiteeye` the 2026-08 one.
    # Both are kept so already-extracted frames keep resolving.
    "whiteeye": FolderLabel("Zosterops virens", resolved=True),
    "capewhiteeye": FolderLabel("Zosterops virens", resolved=True),
    "capebulbul": FolderLabel("Pycnonotus capensis", resolved=True),
    "drongo": FolderLabel("Dicrurus adsimilis", resolved=True),
    # --- negatives, not birds ---------------------------------------------------
    # Hands refilling the feeder. Filed under `other_animal` rather than
    # `obstruction` because a hand is a living thing moving at the port, which is
    # what the class has to generalise; `obstruction` is reserved for the lens
    # being blocked by a leaf or branch. The observer recorded these specifically
    # as hard negatives for the open-set gate, and they are the first in-domain
    # non-bird data the project has had -- A25/A27 note the negative classes have
    # never had a single positive example.
    "hands": FolderLabel("other_animal", resolved=True, note="Human hands at the feeder."),
    # Clips whose filename flags more than one bird in frame. Kept separate so
    # they never silently become single-label training data.
    "multibird": FolderLabel(
        "MULTIPLE",
        resolved=False,
        note="Two or more birds in frame; per-frame labels required before training use.",
    ),
}

UNCUT = "uncut"


@dataclass
class FieldFrame:
    """One extracted frame and everything needed to group or split it."""

    path: str
    folder: str
    label: str | None  # None for uncut; may be a species OR a coarser group
    label_resolved: bool  # False = the label is a group, not a species
    session: str  # yyyymmdd_hhmmss -- the grouping key
    clip: str
    t_seconds: float


def _session_of(filename: str) -> str:
    """Recording session id from the filename stem.

    Files are named `<yyyymmdd>_<hhmmss>[_<cut>].mp4`; every cut sharing the
    leading timestamp came out of one continuous recording.
    """
    m = re.match(r"(\d{8}_\d{6})", filename)
    return m.group(1) if m else "unknown"


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def _drop_unreadable(paths: list[Path]) -> list[Path]:
    """Remove frames PIL cannot open, deleting them from disk.

    ffmpeg occasionally emits a zero-byte JPEG at a clip boundary -- one in
    11,051 on the first real run. That single file killed an overnight job
    eight hours before anyone could see it, because the failure surfaced in a
    DataLoader worker rather than here. One bad frame out of eleven thousand
    must cost one frame, not the run, so the check belongs at the point of
    writing where the file can simply be discarded.
    """
    from PIL import Image

    good = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            good.append(p)
        except Exception as exc:  # noqa: BLE001 -- any unreadable file is equally useless
            logger.warning("  dropping unreadable frame %s (%s)", p.name, type(exc).__name__)
            p.unlink(missing_ok=True)
    return good


def extract_clip(clip: Path, out_dir: Path, fps: float, short_side: int) -> list[Path]:
    """Decode one clip to JPEGs at `fps`, scaled to match corpus preprocessing.

    Returns readable paths in temporal order. Idempotent: a clip whose frames
    already exist is skipped, so an interrupted run resumes rather than
    re-decoding hours of video. Previously-extracted frames are re-validated on
    the skip path too, so a run that failed on a bad frame is repaired simply
    by running it again.
    """
    stem = clip.stem
    # Five explicit digit places, NOT `{stem}_*.jpg`. Clip stems are prefixes of
    # one another -- `20260816_143610` is a prefix of `20260816_143610_1` -- so
    # the loose glob makes the base clip claim its sub-clips' frames. That
    # produced 204 duplicate index entries with wrong clip attribution, and
    # could let a clip mistake another's output for its own and skip extraction
    # entirely.
    frame_glob = f"{stem}_[0-9][0-9][0-9][0-9][0-9].jpg"
    existing = sorted(out_dir.glob(frame_glob))
    if existing:
        good = _drop_unreadable(existing)
        logger.info("  %s: %d frames already present, skipping", clip.name, len(good))
        return good

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{stem}_%05d.jpg")
    cmd = [
        "ffmpeg", "-v", "error", "-threads", "4", "-i", str(clip),
        # `scale=-1:{short_side}` pins HEIGHT, which is only the short side for
        # landscape input. 19% of the 2026-09 footage is portrait (1080x1920),
        # including 12 of 25 Amethyst Sunbird clips, and those came out
        # 144x256 -- 44% under-resolution, silently, because the eval transform
        # then upscales back to 224 and recovers no detail.
        # force_original_aspect_ratio=increase pins the SHORT side either way.
        "-vf", f"fps={fps},scale=w={short_side}:h={short_side}"
               ":force_original_aspect_ratio=increase",
        "-q:v", "3", "-y", pattern,
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("  %s: ffmpeg failed: %s", clip.name, r.stderr[-300:])
        return []
    written = _drop_unreadable(sorted(out_dir.glob(frame_glob)))
    logger.info("  %s: %d frames", clip.name, len(written))
    return written


def extract_all(cfg, cut_fps: float = 2.0, uncut_fps: float = 1.0, short_side: int = 256):
    """Extract every clip under ml/training/ and return the frame index."""
    root = cfg.root / "training"
    if not root.is_dir():
        raise FileNotFoundError(f"no field footage at {root}")

    out_root = cfg.path("data_root") / "field" / "frames"
    frames: list[FieldFrame] = []

    for folder_dir in sorted(root.iterdir()):
        if not folder_dir.is_dir():
            continue
        folder = folder_dir.name
        if folder not in FOLDER_TO_LABEL and folder != UNCUT:
            logger.warning("skipping unrecognised folder %r -- add it to FOLDER_TO_LABEL", folder)
            continue

        fps = uncut_fps if folder == UNCUT else cut_fps
        fl = FOLDER_TO_LABEL.get(folder)
        if fl is None:
            shown = "unlabelled"
        else:
            shown = display(fl.label)
            if not fl.resolved:
                shown += " -- GROUP, not a species"
        logger.info("%s (%s) at %.1f fps", folder, shown, fps)

        for clip in sorted(folder_dir.glob("*.mp4")):
            written = extract_clip(clip, out_root / folder, fps, short_side)
            for i, p in enumerate(written):
                frames.append(
                    FieldFrame(
                        path=str(p.relative_to(cfg.root)),
                        folder=folder,
                        label=fl.label if fl else None,
                        label_resolved=bool(fl and fl.resolved),
                        session=_session_of(clip.name),
                        clip=clip.stem,
                        t_seconds=round(i / fps, 2),
                    )
                )

    index = cfg.path("data_root") / "field" / "frames.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps([asdict(f) for f in frames], indent=1), encoding="utf-8")
    logger.info("wrote %s (%d frames)", index, len(frames))
    return frames


def predict(cfg, frames, checkpoint: str = "student_best.pt", batch_size: int = 32):
    """Run the fine-tuned checkpoint over the extracted frames."""
    import numpy as np

    from birdcam.eval.extract import extract as run_extract
    from birdcam.eval.extract import load_model_from_checkpoint

    @dataclass
    class _Item:
        path: Path

    model, _ = load_model_from_checkpoint(cfg, cfg.path("checkpoints_dir") / checkpoint)
    items = [_Item(path=cfg.root / f.path) for f in frames]
    feats, taxon_logits, sex_logits = run_extract(cfg, model, items, batch_size=batch_size)

    out = cfg.path("data_root") / "field" / "predictions.npz"
    np.savez_compressed(
        out,
        paths=np.array([f.path for f in frames], dtype=object),
        folders=np.array([f.folder for f in frames], dtype=object),
        sessions=np.array([f.session for f in frames], dtype=object),
        clips=np.array([f.clip for f in frames], dtype=object),
        features=feats,
        taxon_logits=taxon_logits,
        sex_logits=sex_logits,
        taxon_classes=np.array(cfg.taxon_classes, dtype=object),
        checkpoint=checkpoint,
    )
    logger.info("wrote %s", out)
    return out


def main() -> None:
    import argparse

    from birdcam.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut-fps", type=float, default=2.0)
    ap.add_argument("--uncut-fps", type=float, default=1.0)
    ap.add_argument("--checkpoint", default="student_best.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    frames = extract_all(cfg, cut_fps=args.cut_fps, uncut_fps=args.uncut_fps)
    print(f"extracted {len(frames)} frames")
    if not args.extract_only:
        predict(cfg, frames, checkpoint=args.checkpoint, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
