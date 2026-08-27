"""Rendering a model slug as a dashboard row, without overclaiming."""

from __future__ import annotations

import pytest

from capture.labels import display_name, scientific_name


class FakeSpecies:
    def __init__(self, slug, common, sci):
        self.slug, self.common_name, self.scientific_name = slug, common, sci


class FakeConfig:
    species = [
        FakeSpecies("cinnyris_chalybeus", "Southern Double-collared Sunbird", "Cinnyris chalybeus"),
        FakeSpecies("promerops_cafer", "Cape Sugarbird", "Promerops cafer"),
    ]


def test_species_slug_becomes_its_common_name():
    assert display_name("cinnyris_chalybeus", FakeConfig()) == "Southern Double-collared Sunbird"


def test_genus_fallback_never_names_a_species():
    """`cinnyris_indet` stands for a pair that cannot be separated in the field.

    data/field.py: a label deliberately coarser than a species must not be able
    to quietly become a species later.
    """
    assert display_name("cinnyris_indet", FakeConfig()) == "Cinnyris sp."
    assert scientific_name("cinnyris_indet", FakeConfig()) is None


def test_family_fallback_renders_as_a_family():
    assert display_name("nectariniidae_indet") == "Nectariniidae sp."


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("nectarivore_indet", "Nectarivore (species uncertain)"),
        ("non_target_indet", "Non-target bird"),
    ],
)
def test_guild_labels_are_not_rendered_as_taxa(label, expected):
    """A guild is a functional grouping, not a rank. 'Nectarivore sp.' is wrong."""
    assert display_name(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("empty_feeder", "Empty feeder"),
        ("insect", "Insect"),
        ("unknown", "Unknown (not recognised)"),
        ("uncertain", "Uncertain"),
    ],
)
def test_negatives_and_sentinels_render_readably(label, expected):
    assert display_name(label) == expected


def test_unknown_slug_without_config_falls_back_to_the_binomial():
    assert display_name("zosterops_virens") == "Zosterops virens"


def test_scientific_name_for_a_known_species():
    assert scientific_name("promerops_cafer", FakeConfig()) == "Promerops cafer"
