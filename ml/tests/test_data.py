"""Data-layer tests: manifest integrity, dedup, label mapping, split leakage.

The split-leakage tests are the important ones. Every metric this project
produces is conditional on the test set being genuinely unseen; if a burst of
near-identical photos straddles a split boundary, the numbers are fiction and
nothing downstream can be trusted.

Tests that need a populated manifest skip cleanly when one is absent, so the
suite still runs on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdcam.config import load_config
from birdcam.data.dataset import LabelMapper, load_labelled
from birdcam.data.manifest import Manifest, open_manifest


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def manifest(cfg):
    path = cfg.path("manifest_db")
    if not path.is_file():
        pytest.skip("no manifest; run the fetcher first")
    with open_manifest(path) as m:
        if m.count(status="downloaded") == 0:
            pytest.skip("manifest has no downloaded images")
        yield m


# --- label mapping (no manifest needed) --------------------------------------


def test_male_maps_to_masked_pair_not_breeding(cfg):
    """An annotated male must NOT be assigned a plumage state.

    No public source annotates eclipse vs breeding, so the target is the
    admissible set {male_breeding, male_eclipse}.
    """
    mapper = LabelMapper(cfg)
    mask, name = mapper.sex_target("Cinnyris chalybeus", "Male", "Adult")
    assert name == "male_unspecified"
    assert mask.sum() == 2
    assert mask[cfg.sex_class_index["male_breeding"]] == 1.0
    assert mask[cfg.sex_class_index["male_eclipse"]] == 1.0


def test_female_maps_to_exact_single_label(cfg):
    mapper = LabelMapper(cfg)
    mask, name = mapper.sex_target("Cinnyris chalybeus", "Female", "Adult")
    assert name == "female"
    assert mask.sum() == 1
    assert mask[cfg.sex_class_index["female"]] == 1.0


def test_unannotated_becomes_indeterminate_never_dropped(cfg):
    """Images with no sex annotation belong in `indeterminate`, not the bin."""
    mapper = LabelMapper(cfg)
    mask, name = mapper.sex_target("Cinnyris chalybeus", None, None)
    assert name == "indeterminate"
    assert mask[cfg.sex_class_index["indeterminate"]] == 1.0


def test_juvenile_takes_precedence_over_sex(cfg):
    """Juvenile plumage is what the model actually sees."""
    mapper = LabelMapper(cfg)
    _, name = mapper.sex_target("Cinnyris chalybeus", "Male", "Juvenile")
    assert name == "juvenile"


def test_monomorphic_species_forced_to_not_applicable(cfg):
    """Training the sex head on a monomorphic species is training on noise."""
    mapper = LabelMapper(cfg)
    _, name = mapper.sex_target("Zosterops virens", "Male", "Adult")
    assert name == "not_applicable"


def test_cannot_be_determined_is_indeterminate(cfg):
    mapper = LabelMapper(cfg)
    _, name = mapper.sex_target("Cinnyris chalybeus", "Cannot Be Determined", None)
    assert name == "indeterminate"


def test_every_mask_has_at_least_one_admissible_class(cfg):
    """A sample with an all-zero mask would produce -inf loss."""
    mapper = LabelMapper(cfg)
    for sex in (None, "Male", "Female", "Cannot Be Determined"):
        for life in (None, "Adult", "Juvenile"):
            mask, _ = mapper.sex_target("Cinnyris chalybeus", sex, life)
            assert mask.sum() >= 1.0, (sex, life)


# --- manifest integrity -------------------------------------------------------


def test_every_image_has_an_allowed_licence(cfg, manifest: Manifest):
    """CC-BY and CC-BY-NC require attribution; no licence means unusable."""
    allowed = set(cfg.train_cfg["fetch"]["allowed_licenses"])
    bad = [
        r["image_id"]
        for r in manifest.iter_rows("status='downloaded'")
        if (r["license"] or "").lower() not in allowed
    ]
    assert not bad, f"{len(bad)} images carry a disallowed licence, e.g. {bad[:3]}"


def test_every_image_has_attribution(cfg, manifest: Manifest):
    missing = [
        r["image_id"]
        for r in manifest.iter_rows("status='downloaded'")
        if not r["attribution_text"]
    ]
    assert not missing, f"{len(missing)} images lack attribution text"


def test_downloaded_images_have_hash_and_path(cfg, manifest: Manifest):
    bad = [
        r["image_id"]
        for r in manifest.iter_rows("status='downloaded'")
        if not r["sha256"] or not r["local_path"]
    ]
    assert not bad, f"{len(bad)} downloaded rows missing sha256/local_path"


def test_image_ids_are_unique(cfg, manifest: Manifest):
    ids = [r["image_id"] for r in manifest.iter_rows()]
    assert len(ids) == len(set(ids))


# --- dedup --------------------------------------------------------------------


def test_no_two_kept_images_share_a_phash(cfg, manifest: Manifest):
    """Exact perceptual-hash collisions must have been collapsed."""
    seen: dict[str, str] = {}
    for r in manifest.iter_rows("status='downloaded' AND phash IS NOT NULL"):
        if r["phash"] in seen:
            pytest.fail(f"duplicate phash kept: {r['image_id']} and {seen[r['phash']]}")
        seen[r["phash"]] = r["image_id"]


def test_duplicates_are_excluded_from_splits(cfg, manifest: Manifest):
    leaked = [r["image_id"] for r in manifest.iter_rows("status='duplicate' AND split IS NOT NULL")]
    assert not leaked, f"{len(leaked)} duplicates were assigned a split"


# --- split leakage: the tests that make the metrics meaningful ----------------


def test_no_observation_appears_in_two_splits(cfg, manifest: Manifest):
    """A burst of near-identical frames straddling a split makes metrics fiction."""
    obs_splits: dict[str, set[str]] = {}
    for r in manifest.iter_rows("split IS NOT NULL AND observation_id IS NOT NULL"):
        obs_splits.setdefault(r["observation_id"], set()).add(r["split"])
    straddling = {o: s for o, s in obs_splits.items() if len(s) > 1}
    assert not straddling, (
        f"{len(straddling)} observations span multiple splits, e.g. {list(straddling.items())[:3]}"
    )


def test_no_observer_species_group_spans_splits(cfg, manifest: Manifest):
    """The declared group key must actually hold.

    Photographer style (camera, garden, processing) recurs across an observer's
    images; letting it cross the boundary lets the model learn the photographer.
    """
    groups: dict[tuple[str, str], set[str]] = {}
    for r in manifest.iter_rows("split IS NOT NULL"):
        key = (r["observer_id"] or f"obs:{r['observation_id']}", r["scientific_name"])
        groups.setdefault(key, set()).add(r["split"])
    straddling = {k: v for k, v in groups.items() if len(v) > 1}
    assert not straddling, f"{len(straddling)} groups span splits, e.g. {list(straddling)[:3]}"


def test_all_three_splits_are_populated(cfg, manifest: Manifest):
    counts = {s: manifest.count(split=s, status="downloaded") for s in ("train", "val", "test")}
    assert all(v > 0 for v in counts.values()), counts


def test_split_proportions_are_near_target(cfg, manifest: Manifest):
    """Grouped assignment cannot hit targets exactly; it must not be wild either."""
    target = cfg.train_cfg["preprocess"]["split"]
    counts = {s: manifest.count(split=s, status="downloaded") for s in ("train", "val", "test")}
    total = sum(counts.values())
    for s, n in counts.items():
        assert abs(n / total - target[s]) < 0.06, f"{s}: {n / total:.3f} vs {target[s]}"


def test_every_species_present_in_test_split(cfg, manifest: Manifest):
    """A species with no test images cannot be evaluated at all."""
    in_test = {
        r["scientific_name"] for r in manifest.iter_rows("split='test' AND status='downloaded'")
    }
    in_train = {
        r["scientific_name"] for r in manifest.iter_rows("split='train' AND status='downloaded'")
    }
    missing = in_train - in_test
    assert not missing, f"species trained but never evaluated: {sorted(missing)}"


# --- labelled loading ---------------------------------------------------------


def test_labelled_images_align_with_manifest(cfg, manifest: Manifest):
    items = load_labelled(cfg, manifest)
    assert items
    assert all(i.taxon_index < len(cfg.taxon_classes) for i in items)
    assert all(i.sex_mask.shape == (len(cfg.sex_classes),) for i in items)
    assert all(i.sex_mask.sum() >= 1 for i in items)


def test_no_image_appears_twice_in_labelled_set(cfg, manifest: Manifest):
    items = load_labelled(cfg, manifest)
    ids = [i.image_id for i in items]
    assert len(ids) == len(set(ids))


def test_embedding_cache_aligns_if_present(cfg):
    """Feature/label misalignment yields plausible metrics that are pure noise."""
    d = cfg.path("embeddings_dir")
    if not d.is_dir():
        pytest.skip("no embedding cache")
    npys = sorted(d.glob("*.npy"))
    if not npys:
        pytest.skip("no embedding cache")
    from birdcam.train.cache_embeddings import load_cached

    feats, items = load_cached(cfg, npys[0].stem)
    assert len(feats) == len(items)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any(), "NaNs in cached features"
