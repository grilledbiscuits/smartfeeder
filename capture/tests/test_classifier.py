"""Preprocessing, the artefact-pairing guard, and the novelty seam.

Nothing here loads a model. The pieces under test are the ones that decide
whether a model is safe to use at all, plus the pixel path that has to match
`birdcam.train.augment.build_eval_transform` exactly or every logit is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from capture.classifier import (
    ClassifierUnavailable,
    build_novelty_scorer,
    check_artefact_pairing,
    load_range_prior,
    preprocess_frame,
)

FINE_TUNED = {
    "source": "fine-tuned checkpoint student_best.pt @ a71e95cca471",
    "temperature": 0.986,
}


def test_frozen_export_paired_with_finetuned_calibration_is_rejected():
    """ASSUMPTIONS.md A26, verbatim: these are two different models.

    Both artefacts are individually valid, which is what makes the pairing
    dangerous -- nothing fails, the numbers are simply wrong.
    """
    sidecar = {
        "trained": True,
        "training": "frozen-feature linear heads, standardisation folded in",
    }
    problems = check_artefact_pairing(sidecar, FINE_TUNED)
    assert len(problems) == 1
    assert "different models" in problems[0]
    assert "A26" in problems[0]


def test_matching_artefacts_pass():
    sidecar = {"trained": True, "training": "full fine-tune from student_best.pt"}
    assert check_artefact_pairing(sidecar, FINE_TUNED) == []


def test_untrained_graph_is_rejected():
    """`to_onnx.export` will happily export random weights; they must not ship."""
    problems = check_artefact_pairing({"trained": False, "training": "random"}, FINE_TUNED)
    assert any("randomly-initialised" in p for p in problems)


def test_expect_source_mismatch_is_reported():
    sidecar = {"trained": True, "training": "full fine-tune"}
    problems = check_artefact_pairing(sidecar, FINE_TUNED, expect_source="deadbeef")
    assert any("expect_source" in p for p in problems)


def test_expect_source_match_passes():
    sidecar = {"trained": True, "training": "full fine-tune"}
    assert check_artefact_pairing(sidecar, FINE_TUNED, expect_source="a71e95cca471") == []


# -- preprocessing -------------------------------------------------------------


def make_jpeg(path, size=(640, 480), colour=(120, 90, 60)):
    from PIL import Image

    Image.new("RGB", size, colour).save(path)
    return path


def test_preprocess_produces_a_normalised_chw_tensor(tmp_path):
    arr = preprocess_frame(make_jpeg(tmp_path / "f.jpg"), 224)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32


def test_preprocess_applies_imagenet_statistics(tmp_path):
    """A mid-grey frame must land near the ImageNet mean, not near zero."""
    grey = make_jpeg(tmp_path / "g.jpg", colour=(124, 116, 104))  # ~ the mean
    arr = preprocess_frame(grey, 224)
    assert np.allclose(arr.mean(axis=(1, 2)), 0.0, atol=0.05)


def test_preprocess_scales_the_short_side_then_centre_crops(tmp_path):
    """Matches build_eval_transform: Resize(int(size*1.14)) then CenterCrop.

    A portrait and a landscape frame of the same scene must both survive as a
    square centre crop rather than being squashed.
    """
    for size in [(640, 480), (480, 640), (1280, 720)]:
        arr = preprocess_frame(make_jpeg(tmp_path / f"f{size[0]}x{size[1]}.jpg", size), 224)
        assert arr.shape == (3, 224, 224)


# -- the novelty seam ----------------------------------------------------------


def test_disabled_novelty_returns_none(caplog):
    """The current state: no gate, and the log must say so loudly."""
    with caplog.at_level("WARNING"):
        assert build_novelty_scorer({"enabled": False}, graph_emits_features=False) is None
    assert "OPEN-SET FAILSAFE DISABLED" in caplog.text


def test_energy_scorer_works_with_a_logits_only_graph():
    """The option available today: energy needs no features."""
    scorer = build_novelty_scorer(
        {"enabled": True, "scorer": "energy", "threshold": -8.4859},
        graph_emits_features=False,
    )
    assert scorer.threshold == pytest.approx(-8.4859)
    # Confident logits are low-energy; flat logits are high-energy and novel.
    confident = scorer.score(None, np.array([[10.0, 0.0, 0.0]]))
    flat = scorer.score(None, np.array([[0.1, 0.0, 0.05]]))
    assert confident[0] < flat[0]


def test_knn_scorer_is_refused_without_features():
    """A25/A26: the graph emits logits only, so kNN cannot be reconstructed."""
    with pytest.raises(ClassifierUnavailable, match="taxon_logits and"):
        build_novelty_scorer(
            {"enabled": True, "scorer": "knn", "threshold": 0.5, "reference": "x.npz"},
            graph_emits_features=False,
        )


def test_enabled_novelty_without_a_threshold_is_refused():
    with pytest.raises(ClassifierUnavailable, match="operational choice"):
        build_novelty_scorer({"enabled": True, "scorer": "energy"}, graph_emits_features=True)


# -- range prior ---------------------------------------------------------------


def test_range_prior_loads_site_weights(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text("weights:\n  cinnyris_chalybeus: 0.9571\n  cinnyris_afer: 0.0591\n")
    weights = load_range_prior(site)
    assert weights["cinnyris_chalybeus"] == pytest.approx(0.9571)


def test_no_site_prior_is_an_empty_dict():
    assert load_range_prior(None) == {}


def test_empty_weights_block_warns(tmp_path, caplog):
    """A21: nothing currently tells the model what a Cape Town feeder sees."""
    site = tmp_path / "site.yaml"
    site.write_text("weights: {}\n")
    with caplog.at_level("WARNING"):
        assert load_range_prior(site) == {}
    assert "A21" in caplog.text
