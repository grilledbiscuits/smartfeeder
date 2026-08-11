"""Phase 1 tests: config loading and label-space construction.

These target the failure modes that would otherwise produce a model that trains
happily and means nothing -- a species with no rollup parent, a sex annotation
mapped to a class that does not exist, thresholds ordered backwards.
"""

from __future__ import annotations

import copy

import pytest

from birdcam.config import Config, ConfigError, load_config


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


# --- species list -------------------------------------------------------------


def test_all_three_tiers_present(cfg: Config) -> None:
    assert {s.tier for s in cfg.species} == {"A", "B", "C"}


def test_tier_counts_match_brief(cfg: Config) -> None:
    """Guards against a species being dropped during a config edit."""
    assert len(cfg.species_by_tier("A")) == 6
    assert len(cfg.species_by_tier("B")) == 13
    assert len(cfg.species_by_tier("C")) == 18


def test_tier_c_is_present_and_nonempty(cfg: Config) -> None:
    """Tier C is mandatory, not optional.

    Without confusable non-target species the model forces every visitor into a
    sunbird class.
    """
    assert len(cfg.species_by_tier("C")) > 0


def test_species_slugs_are_unique(cfg: Config) -> None:
    slugs = [s.slug for s in cfg.species]
    assert len(slugs) == len(set(slugs))


def test_genus_derives_from_binomial(cfg: Config) -> None:
    assert cfg.species_by_name["Cinnyris chalybeus"].genus == "Cinnyris"
    assert cfg.species_by_name["Cinnyris chalybeus"].slug == "cinnyris_chalybeus"


def test_rare_tier_b_species_fetched_globally(cfg: Config) -> None:
    """Measured 2026-07-31: these have zero research-grade ZA records.

    A ZA-restricted fetch would return nothing for them.
    """
    for name in ("Cinnyris manoensis", "Cinnyris cupreus"):
        assert cfg.species_by_name[name].fetch_scope == "global"


# --- taxon label space --------------------------------------------------------


def test_taxon_class_order_is_deterministic(cfg: Config) -> None:
    """Logit index meaning must be stable across runs and machines."""
    assert load_config().taxon_classes == cfg.taxon_classes


def test_every_species_has_a_taxon_class(cfg: Config) -> None:
    for s in cfg.species:
        assert s.slug in cfg.taxon_class_index


def test_fallback_and_negative_classes_present(cfg: Config) -> None:
    for label in (
        "cinnyris_indet",
        "promerops_indet",
        "nectariniidae_indet",
        "nectarivore_indet",
        "empty_feeder",
        "insect",
        "other_animal",
        "obstruction",
    ):
        assert label in cfg.taxon_class_index, label


def test_promerops_rolls_up_to_promeropidae_not_nectariniidae(cfg: Config) -> None:
    """Verified against GBIF 2026-07-31.

    Promerops cafer is Promeropidae. Rolling the Cape Sugarbird up into
    nectariniidae_indet would be taxonomically wrong.
    """
    assert cfg.genus_to_family["Promerops"] == "Promeropidae"
    assert cfg.genus_to_family["Cinnyris"] == "Nectariniidae"


def test_nectarivore_guild_spans_both_families(cfg: Config) -> None:
    """The capture app's real question spans the taxonomic split."""
    assert cfg.family_to_guild["Nectariniidae"] == "nectarivore"
    assert cfg.family_to_guild["Promeropidae"] == "nectarivore"
    assert cfg.family_to_guild["Ploceidae"] == "non_target"


# --- sex / plumage head -------------------------------------------------------


def test_sex_classes_match_schema(cfg: Config) -> None:
    assert cfg.sex_classes == [
        "male_breeding",
        "male_eclipse",
        "female",
        "juvenile",
        "indeterminate",
        "not_applicable",
    ]


def test_male_is_a_masked_partial_label(cfg: Config) -> None:
    """No source distinguishes breeding from eclipse plumage.

    Verified against the live iNaturalist controlled-terms API 2026-07-31: the
    Sex term has exactly Male / Female / Cannot Be Determined. Annotated males
    must therefore train the summed group mass, not a fabricated
    `male_breeding` label.
    """
    groups = cfg.partial_label_groups
    assert groups["male_unspecified"] == ["male_breeding", "male_eclipse"]


def test_inat_male_maps_to_masked_group_not_breeding(cfg: Config) -> None:
    mapping = cfg.taxonomy_cfg["sex_plumage_head"]["annotation_mapping"]["inaturalist"]
    assert mapping["9|11"] == "male_unspecified"
    assert mapping["9|11"] != "male_breeding", "would fabricate a plumage-state label"
    assert mapping["9|10"] == "female"


def test_cannot_be_determined_is_a_real_label(cfg: Config) -> None:
    """`indeterminate` is a first-class prediction, not a failure."""
    mapping = cfg.taxonomy_cfg["sex_plumage_head"]["annotation_mapping"]["inaturalist"]
    assert mapping["9|20"] == "indeterminate"


def test_unannotated_images_are_kept_as_indeterminate(cfg: Config) -> None:
    """The pipeline must never silently drop images lacking a sex annotation."""
    assert cfg.taxonomy_cfg["sex_plumage_head"]["unannotated_label"] == "indeterminate"


# --- rollup -------------------------------------------------------------------


def test_rollup_thresholds_increase_with_generality(cfg: Config) -> None:
    th = cfg.taxonomy_cfg["rollup"]["thresholds"]
    assert th["species"] <= th["genus"] <= th["family"] <= th["guild"]


# --- validation catches real mistakes ----------------------------------------


def _mutated(cfg: Config, **overrides) -> Config:
    """Build an unvalidated Config with deep-copied sections."""
    return Config(
        species_cfg=overrides.get("species_cfg", copy.deepcopy(cfg.species_cfg)),
        taxonomy_cfg=overrides.get("taxonomy_cfg", copy.deepcopy(cfg.taxonomy_cfg)),
        train_cfg=overrides.get("train_cfg", copy.deepcopy(cfg.train_cfg)),
        root=cfg.root,
    )


def test_validate_rejects_species_without_genus_fallback(cfg: Config) -> None:
    tax = copy.deepcopy(cfg.taxonomy_cfg)
    del tax["taxon_head"]["genus_fallback"]["cinnyris_indet"]
    with pytest.raises(ConfigError, match="Cinnyris"):
        _mutated(cfg, taxonomy_cfg=tax).validate()


def test_validate_rejects_dead_single_species_genus_class(cfg: Config) -> None:
    """A genus with one species gives the model nothing to be uncertain between."""
    tax = copy.deepcopy(cfg.taxonomy_cfg)
    tax["taxon_head"]["genus_fallback"]["nectarinia_indet"] = {"genus": "Nectarinia"}
    with pytest.raises(ConfigError, match="untrainable dead class"):
        _mutated(cfg, taxonomy_cfg=tax).validate()


def test_validate_rejects_multi_species_genus_without_fallback(cfg: Config) -> None:
    """Dropping cinnyris_indet would remove the single most important
    uncertainty this project needs to express."""
    tax = copy.deepcopy(cfg.taxonomy_cfg)
    del tax["taxon_head"]["genus_fallback"]["cinnyris_indet"]
    with pytest.raises(ConfigError, match="unable to express"):
        _mutated(cfg, taxonomy_cfg=tax).validate()


def test_every_genus_has_a_family(cfg: Config) -> None:
    for s in cfg.species:
        assert s.genus in cfg.genus_to_family, s.genus


def test_single_species_genera_have_no_genus_class(cfg: Config) -> None:
    declared = {g["genus"] for g in cfg.taxonomy_cfg["taxon_head"]["genus_fallback"].values()}
    for genus, members in cfg.species_per_genus.items():
        if len(members) == 1:
            assert genus not in declared, f"{genus} would be a dead class"


def test_validate_rejects_annotation_mapped_to_unknown_class(cfg: Config) -> None:
    tax = copy.deepcopy(cfg.taxonomy_cfg)
    tax["sex_plumage_head"]["annotation_mapping"]["inaturalist"]["9|11"] = "male_nonexistent"
    with pytest.raises(ConfigError, match="neither a sex class nor"):
        _mutated(cfg, taxonomy_cfg=tax).validate()


def test_validate_rejects_backwards_thresholds(cfg: Config) -> None:
    tax = copy.deepcopy(cfg.taxonomy_cfg)
    tax["rollup"]["thresholds"]["genus"] = 0.1
    with pytest.raises(ConfigError, match="must increase with generality"):
        _mutated(cfg, taxonomy_cfg=tax).validate()


def test_validate_rejects_bad_split_fractions(cfg: Config) -> None:
    tr = copy.deepcopy(cfg.train_cfg)
    tr["preprocess"]["split"]["train"] = 0.9
    with pytest.raises(ConfigError, match="Split fractions"):
        _mutated(cfg, train_cfg=tr).validate()


def test_validate_rejects_duplicate_species(cfg: Config) -> None:
    sp = copy.deepcopy(cfg.species_cfg)
    sp["tiers"]["A"]["species"].append(
        {"scientific_name": "Cinnyris chalybeus", "common_name": "dupe"}
    )
    with pytest.raises(ConfigError, match="more than once"):
        _ = _mutated(cfg, species_cfg=sp).species


def test_validate_rejects_monomorphic_typo(cfg: Config) -> None:
    tax = copy.deepcopy(cfg.taxonomy_cfg)
    tax["sex_plumage_head"]["monomorphic_forced_na"].append("Zosterops typo")
    with pytest.raises(ConfigError, match="not in species.yaml"):
        _mutated(cfg, taxonomy_cfg=tax).validate()


# --- environment --------------------------------------------------------------


def test_contact_email_required_for_fetching(cfg: Config, monkeypatch) -> None:
    """Fetchers must refuse to run without a contact address."""
    monkeypatch.delenv("BIRDCAM_CONTACT", raising=False)
    with pytest.raises(ConfigError, match="contact email"):
        cfg.contact_email()


def test_user_agent_includes_contact(cfg: Config, monkeypatch) -> None:
    monkeypatch.setenv("BIRDCAM_CONTACT", "someone@example.com")
    ua = cfg.user_agent()
    assert "someone@example.com" in ua
    assert "birdcam" in ua


def test_paths_are_relative_to_repo_root(cfg: Config) -> None:
    """No absolute paths in config; everything resolves under the repo."""
    assert cfg.path("manifest_db").is_relative_to(cfg.root)
    with pytest.raises(ConfigError, match="Unknown path key"):
        cfg.path("nope")
