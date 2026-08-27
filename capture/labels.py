"""Turning a model label into something a dashboard row can show.

The classifier emits slugs (`cinnyris_chalybeus`, `cinnyris_indet`,
`uncertain`); `web.visits.species` is free text and the existing rows use
common names ("Southern Double-collared Sunbird"). Something has to bridge
them, and it matters that the bridge does not overclaim.

`data/field.py` makes the case at length: a label that is deliberately coarser
than a species must not be able to quietly become a species later. So
`cinnyris_indet` renders as "Cinnyris sp." -- an honest genus-level answer --
and never as one of the two double-collared sunbirds it stands for.

Common names come from `config/species.yaml` via the loaded birdcam Config, so
this module holds no species list of its own.
"""

from __future__ import annotations

from typing import Any

# Fallback slugs carry a suffix rather than being enumerated, so a new family
# or guild node in taxonomy.yaml renders correctly with no edit here.
_INDET_SUFFIX = "_indet"

_NEGATIVE_NAMES = {
    "empty_feeder": "Empty feeder",
    "insect": "Insect",
    "other_animal": "Other animal",
    "obstruction": "Obstruction",
}

_SENTINEL_NAMES = {
    "unknown": "Unknown (not recognised)",
    "uncertain": "Uncertain",
}

# Guild rollup labels. `Classifier.decide` builds these as f"{guild}_indet" from
# the guild values in taxonomy.yaml's family_fallback, so they are not taxon
# names and must not be rendered as "<Genus> sp.". Note that only
# `nectarivore_indet` exists as an actual class -- see the note in
# capture/README.md about the guild rollup emitting `non_target_indet`, which
# is not in the label space at all.
_GUILD_NAMES = {
    "nectarivore": "Nectarivore (species uncertain)",
    "non_target": "Non-target bird",
}


def display_name(label: str, cfg: Any | None = None) -> str:
    """Human-readable name for one taxon label.

    `cfg` is a `birdcam.config.Config`; when absent (tests, mock mode) the
    slug is still rendered readably, just without common names.
    """
    if not label:
        return "unknown"
    if label in _SENTINEL_NAMES:
        return _SENTINEL_NAMES[label]
    if label in _NEGATIVE_NAMES:
        return _NEGATIVE_NAMES[label]

    if label.endswith(_INDET_SUFFIX):
        stem = label[: -len(_INDET_SUFFIX)]
        # A guild is a functional grouping, not a taxon: "Nectarivore sp." would
        # be wrong in a way an ornithologist would notice.
        if stem in _GUILD_NAMES:
            return _GUILD_NAMES[stem]
        return f"{stem.capitalize()} sp."

    if cfg is not None:
        for species in getattr(cfg, "species", []):
            if species.slug == label:
                return species.common_name

    # A species class with no config to look it up in: render the binomial
    # rather than the slug.
    return label.replace("_", " ").capitalize()


def scientific_name(label: str, cfg: Any | None = None) -> str | None:
    """Binomial for a species label, or None for anything coarser."""
    if cfg is None or label.endswith(_INDET_SUFFIX):
        return None
    for species in getattr(cfg, "species", []):
        if species.slug == label:
            return species.scientific_name
    return None
