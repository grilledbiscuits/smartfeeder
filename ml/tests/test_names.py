"""The display layer must cover the whole label space.

Scientific slugs are correct as keys and useless as reading material: the
person who runs this project reads common names, and a report that says
`cinnyris_chalybeus` where it means "Southern Double-collared Sunbird" costs
them a lookup every time. These tests exist so a newly added class fails here
rather than quietly leaking a slug into a report six weeks later.
"""

from __future__ import annotations

import pytest

from birdcam.config import load_config
from birdcam.names import UNSURE, build_display_map, display


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_every_taxon_class_has_a_display_name(cfg) -> None:
    """The guard. build_display_map raises if any class is unnamed."""
    m = build_display_map(cfg)
    for c in cfg.taxon_classes:
        assert m[c].strip(), f"{c} has an empty display name"


def test_no_display_name_is_a_scientific_slug(cfg) -> None:
    """A leaked slug is the failure this module exists to prevent."""
    m = build_display_map(cfg)
    for c, name in m.items():
        assert "_" not in name, f"{c} renders as {name!r}, which is still a slug"


def test_species_names_come_from_species_yaml(cfg) -> None:
    """Not duplicated into taxonomy.yaml, where the two would drift apart."""
    m = build_display_map(cfg)
    for s in cfg.species:
        assert m[s.slug] == s.common_name


def test_fallbacks_state_their_uncertainty(cfg) -> None:
    """`cinnyris_indet` is not an identification and must not read like one."""
    for c in cfg.taxon_classes:
        if c.endswith("_indet"):
            assert UNSURE in display(c, cfg), f"{c} reads like a firm identification"


def test_negatives_do_not_read_like_a_bird(cfg) -> None:
    assert display("empty_feeder", cfg) == "Empty feeder"
    assert display("unknown", cfg) == "Unrecognised"


def test_accepts_a_scientific_name_too(cfg) -> None:
    """The field-ingest side carries 'Genus species', not a slug."""
    assert display("Chalcomitra amethystina", cfg) == "Amethyst Sunbird"


def test_unknown_label_passes_through(cfg) -> None:
    """Called from logging; a formatting miss must never raise mid-run."""
    assert display("not_a_class", cfg) == "not_a_class"


def test_capture_and_training_render_the_same_name(cfg) -> None:
    """Two renderers, one vocabulary.

    `capture/labels.py` deliberately does not import birdcam -- it ships to the
    Pi and must stay standalone -- so it reads the same `display:` block through
    the Config it is handed. That is fine right up until the two drift and a
    dashboard row disagrees with the evaluation report about what was seen.
    """
    from capture.labels import display_name

    for c in cfg.taxon_classes:
        if c.startswith("nectarivore"):
            continue  # guild rollup: capture words it differently, see _GUILD_NAMES
        assert display_name(c, cfg) == display(c, cfg), c
