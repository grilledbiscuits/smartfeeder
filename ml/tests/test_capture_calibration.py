"""The calibration transform must be identical to the deployed probability path."""

import numpy as np

from birdcam.config import load_config
from birdcam.inference import Classifier, softmax


def test_probability_transform_matches_decision_and_preserves_raw_energy_input():
    cfg = load_config()
    idx = cfg.taxon_class_index["cinnyris_chalybeus"]
    raw = np.full(len(cfg.taxon_classes), -10.0)
    raw[idx] = 8
    raw[cfg.taxon_class_index["empty_feeder"]] = 20
    before = raw.copy()
    clf = Classifier(cfg, temperature=1.3, range_prior={"cinnyris_chalybeus": 0.3})
    transformed = clf.probability_logits(np.stack([raw, raw]))
    assert np.isneginf(transformed[:, cfg.taxon_class_index["empty_feeder"]]).all()
    expected = softmax(transformed[0], 1.3)[idx]
    assert np.isclose(clf.decide(raw).confidence, expected)
    assert np.array_equal(raw, before)
