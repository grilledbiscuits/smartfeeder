"""Field frames as a second training source, alongside the web corpus.

The manifest holds iNaturalist observations: licensed, attributed, one row per
photograph, thousands of photographers. Field frames are none of those things --
they are frames cut from video shot at one feeder, by one person, over a few
weeks. Forcing them into the manifest schema would mean a dozen columns that
mean nothing and one that quietly lies (`observer_id`), so they keep their own
index and arrive as the same `LabelledImage` records the trainer already eats.

## Why they are not worth the same as a web image

There are more field frames than web images, and they carry far less
information. 22,213 frames come from about thirty recording sessions: a handful
of individual birds, a handful of lighting conditions, one feeder, one garden.
Two frames a second apart are nearly the same photograph. Counted equally they
would dominate the loss and the model would learn *this* garden -- which is
half right, since it is the garden it will be deployed in, and half disastrous,
because it is also how you get a model that has memorised one Amethyst Sunbird
rather than learned the species.

`FIELD_WEIGHT` is therefore a deliberate placeholder, not a derived quantity.
It is the number to move first when field performance disappoints, and it is
recorded in the run config so a result can be traced back to it.

## Splits are by session, and that is stricter than it looks

Two clips from one session are the same bird, in the same light, minutes apart;
so are the uncut frames from that session, including the empty-feeder ones cut
from the gaps between visits. A session therefore goes wholly into one split,
even though that means the empty frames from a session follow its birds around.
That is the point: an "empty feeder" frame shares its background, exposure and
tarp position with the bird frames from the same recording, and splitting them
apart would let the model recognise the background rather than the bird.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict

import numpy as np

from birdcam.config import Config
from birdcam.data.dataset import LabelledImage, LabelMapper

logger = logging.getLogger(__name__)

# Fallback only; the real value lives in train.yaml as train.field_weight.
# Placeholder either way -- see the module docstring: chosen, not measured.
FIELD_WEIGHT = 0.3

# Folder labels that are class slugs already rather than binomials.
_DIRECT_CLASSES = {"other_animal", "empty_feeder", "insect", "obstruction"}


# Clip label -> the set of sex classes that label admits. The loss is
# -log(sum of admissible probabilities), so a full set is zero loss and zero
# gradient: that is how "no supervision" is expressed, and it is NOT the same as
# `indeterminate`, which is a positive claim that the bird could not be sexed.
_SEX_FROM_CLIP = {
    # The observer recorded the sex, not the plumage state, so a male is
    # admissible over both male classes -- the same partial-label treatment an
    # iNaturalist "Male" annotation gets.
    "male": ("male_breeding", "male_eclipse"),
    "female": ("female",),
    # Juvenile outranks sex in the head's precedence: juvenile plumage is what
    # the model actually sees.
    "juvenile": ("juvenile",),
    # `pair` and `unsure` are deliberately absent: both mean no supervision.
}

# Substrings the observer puts in a clip name when more than one bird is in
# frame. Stated at the outset -- "there shouldn't be multiple birds in frame
# unless specified in the filename" -- and honoured only for the `multibird`
# FOLDER until 2026-09-10, which left 266 frames of same-species couples
# training as single birds. The taxon label survives that (both birds are the
# same species, so the label is true of the frame); the sex label does not,
# because a male and a female in one frame make no single sex label true.
_MULTI_BIRD_MARKERS = ("couple", "anddc", "funny")


def is_multi_bird(clip: str) -> bool:
    """Whether the clip's own name says it holds more than one bird."""
    stem = clip.split("_", 2)[-1].lower()
    return any(m in stem for m in _MULTI_BIRD_MARKERS)


def load_clip_sex(cfg: Config) -> dict[str, str]:
    """Per-clip sex/age labels, made by the observer who shot the footage.

    Absent file means no sex supervision from field footage, which is a
    degradation rather than an error: the frames still train the taxon head.
    """
    path = cfg.root / "labels" / "field_clip_sex.json"
    if not path.is_file():
        logger.warning("no %s; field frames will carry no sex supervision", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def sex_mask_for(
    cfg: Config,
    mapper: LabelMapper,
    sci: str,
    clip_label: str | None,
    clip: str = "",
):
    """(mask, name) for a field frame, from its clip's label.

    Field footage has no annotator and no EXIF: before these labels existed every
    frame fell through to `indeterminate`, which in this label space asserts that
    the bird COULD NOT be sexed. For 4,499 frames of a strongly dimorphic species
    -- the male has an iridescent green head and a red breast band, the female is
    plain grey-brown -- that was not a missing label but a false one, and it
    taught the head to abstain on the easiest case it will ever see.

    `unsure` returns an all-ones mask: every class admissible, so the loss term is
    exactly zero and the frame contributes nothing to this head while still
    training the taxon head. Some of those clips hold a male and a female at the
    feeder together, where no single label is true of the frame at all.
    """
    n = len(cfg.sex_classes)
    if clip and is_multi_bird(clip) and sci not in mapper.monomorphic:
        # The filename wins over the clip label here. One clip named
        # `...doublecollaredcouple4` came back labelled `male` -- plausibly only
        # one bird was visible in the frames shown -- but if a female is also in
        # frame at any point, `male` is false for those frames. Fifteen frames
        # are not worth that risk on a Tier A species.
        return np.ones(n, dtype=np.float32), "unsupervised"
    if sci in _DIRECT_CLASSES:
        # An empty feeder, or a pair of hands, has no sex. That is not missing
        # information to be masked away -- it is a true statement, and the same
        # one a monomorphic species gets, so the head can learn it.
        mask = np.zeros(n, dtype=np.float32)
        mask[cfg.sex_class_index["not_applicable"]] = 1.0
        return mask, "not_applicable"
    if sci in mapper.monomorphic:
        return mapper.sex_target(sci, None, None)
    members = _SEX_FROM_CLIP.get(clip_label or "")
    if members is None:
        return np.ones(n, dtype=np.float32), "unsupervised"
    mask = np.zeros(n, dtype=np.float32)
    for m in members:
        mask[cfg.sex_class_index[m]] = 1.0
    return mask, (clip_label if len(members) == 1 else "male_unspecified")


def _load_index(cfg: Config) -> list[dict]:
    path = cfg.path("data_root") / "field" / "frames.json"
    if not path.is_file():
        raise FileNotFoundError(f"no field frame index at {path}; run birdcam.data.field first")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_empty(cfg: Config) -> set[str]:
    """Paths the negative miner accepted as showing an empty feeder.

    Absent file means no empty-feeder frames, not an error: mining is optional
    and the rest of the field corpus stands without it.
    """
    path = cfg.path("data_root") / "field" / "empty_candidates.json"
    if not path.is_file():
        logger.warning("no %s; field corpus will contain no empty_feeder frames", path)
        return set()
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k for k, v in d["frames"].items() if v.get("empty")}


def assign_sessions(
    sessions: dict[str, str],
    sizes: dict[str, int],
    cfg: Config,
    seed: int | None = None,
) -> dict[str, str]:
    """Place whole sessions into train/val/test, stratified by label.

    `sessions` maps session id -> the label it mostly carries; `sizes` maps it
    to its frame count. Placement is greedy on FRAMES, not on session count:
    sessions differ in length by more than an order of magnitude (204 frames to
    2,473), so counting sessions equally gave a 75/20/5 split against a 70/15/15
    target. Largest first, into whichever split is furthest below target.
    """
    sp = cfg.train_cfg["preprocess"]["split"]
    seed = seed if seed is not None else cfg.train_cfg["compute"]["seed"]
    targets = {"train": sp["train"], "val": sp["val"], "test": sp["test"]}

    by_label: dict[str, list[str]] = defaultdict(list)
    for sess, label in sessions.items():
        by_label[label].append(sess)

    out: dict[str, str] = {}
    for label, sess_ids in sorted(by_label.items()):
        # Seeded per label, for the same reason preprocess.make_splits is:
        # adding a species must not reshuffle the sessions of another. The
        # shuffle only breaks ties between equally sized sessions.
        rng = random.Random(f"{seed}|field|{label}")
        sess_ids = sorted(sess_ids)
        rng.shuffle(sess_ids)
        sess_ids.sort(key=lambda x: -sizes[x])
        counts = {"train": 0, "val": 0, "test": 0}
        placed = 0
        for sess in sess_ids:
            placed += sizes[sess]
            deficits = {s: targets[s] * placed - counts[s] for s in targets}
            pick = max(deficits, key=lambda s: deficits[s])
            out[sess] = pick
            counts[pick] += sizes[sess]
    return out


def load_field(cfg: Config, weight: float | None = None) -> list[LabelledImage]:
    """Field frames as LabelledImage records, split by session.

    Frames whose folder label is unresolved -- `multibird`, where two birds
    share the frame and no single taxon is true of it -- are excluded. A
    per-frame label would be needed to use them and none exists.
    """
    if weight is None:
        weight = float(cfg.train_cfg["train"].get("field_weight", FIELD_WEIGHT))
    mapper = LabelMapper(cfg)
    clip_sex = load_clip_sex(cfg)
    index = _load_index(cfg)
    empty = _load_empty(cfg)

    # An uncut frame is an empty-feeder negative only if the miner said so;
    # the rest of the uncut footage is unlabelled and is simply not used.
    usable: list[dict] = []
    for rec in index:
        if rec["folder"] == "uncut":
            if rec["path"] in empty:
                usable.append({**rec, "label": "empty_feeder"})
            continue
        if not rec.get("label_resolved"):
            continue
        usable.append(rec)

    # A session's stratum is the label it mostly carries, so that empty-feeder
    # frames do not each count as their own stratum.
    label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in usable:
        label_counts[rec["session"]][rec["label"]] += 1
    sessions = {s: max(c, key=lambda k: c[k]) for s, c in label_counts.items()}
    sizes = {s: sum(c.values()) for s, c in label_counts.items()}
    split_of = assign_sessions(sessions, sizes, cfg)

    out: list[LabelledImage] = []
    skipped: dict[str, int] = defaultdict(int)
    for rec in usable:
        label = rec["label"]
        if label in _DIRECT_CLASSES:
            taxon_label, sci = label, label
        else:
            taxon_label, sci = mapper.taxon_label(label), label
        idx = cfg.taxon_class_index.get(taxon_label)
        if idx is None:
            skipped[taxon_label] += 1
            continue
        mask, name = sex_mask_for(cfg, mapper, sci, clip_sex.get(rec["clip"]), rec["clip"])
        out.append(
            LabelledImage(
                image_id=f"field:{rec['path']}",
                path=cfg.root / rec["path"],
                scientific_name=sci,
                taxon_label=taxon_label,
                taxon_index=idx,
                sex_mask=mask,
                sex_label_name=name,
                split=split_of[rec["session"]],
                observation_id=rec["clip"],
                observer_id=f"field:{rec['session']}",
                source="field",
                weight=weight,
                multi_bird=is_multi_bird(rec["clip"]),
            )
        )
    if skipped:
        logger.warning("field frames with no matching taxon class: %s", dict(skipped))
    return out
