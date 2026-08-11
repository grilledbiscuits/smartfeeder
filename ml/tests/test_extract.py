"""Phase 7 tests: the staleness guard on checkpoint extractions.

The bug this module exists to prevent is silent, not loud. Before 2026-08-05,
`eval/` loaded frozen sweep embeddings validated only by row count, so
evaluating a fine-tuned checkpoint quietly reported the pre-fine-tune model
instead. A features file that is merely *plausible* is worse than one that is
missing, because the numbers it produces look correct.

These tests therefore target the detection of wrongness, not the happy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdcam.config import load_config
from birdcam.eval.extract import _fingerprint, load_extraction


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _write(tmp_path, cfg, image_ids, taxon_classes=None, sha="deadbeef", corrupt=False):
    """Write a minimal extraction file with controllable contents."""
    taxon_classes = taxon_classes or list(cfg.taxon_classes)
    ids = np.array(image_ids, dtype=object)
    fp = _fingerprint(sha, ids, taxon_classes)
    path = tmp_path / "ext.npz"
    np.savez_compressed(
        path,
        image_ids=ids,
        features=np.zeros((len(ids), 4), dtype=np.float32),
        taxon_logits=np.zeros((len(ids), len(taxon_classes)), dtype=np.float32),
        sex_logits=np.zeros((len(ids), len(cfg.sex_classes)), dtype=np.float32),
        taxon_classes=np.array(taxon_classes, dtype=object),
        sex_classes=np.array(list(cfg.sex_classes), dtype=object),
        checkpoint="test.pt",
        checkpoint_sha=sha,
        fingerprint="0" * 64 if corrupt else fp,
    )
    return path


# --- fingerprint sensitivity --------------------------------------------------


def test_fingerprint_detects_swapped_id_at_same_length(cfg) -> None:
    """The failure the old row-count check could not see.

    A manifest that drops one image and gains another keeps its length. Hashing
    the ordered IDs is what makes that detectable.
    """
    a = _fingerprint("sha", np.array(["a", "b", "c"], dtype=object), ["x", "y"])
    b = _fingerprint("sha", np.array(["a", "b", "d"], dtype=object), ["x", "y"])
    assert a != b


def test_fingerprint_detects_reordering(cfg) -> None:
    """Row order is the alignment contract between features and labels."""
    a = _fingerprint("sha", np.array(["a", "b"], dtype=object), ["x"])
    b = _fingerprint("sha", np.array(["b", "a"], dtype=object), ["x"])
    assert a != b


def test_fingerprint_detects_different_checkpoint(cfg) -> None:
    ids = np.array(["a", "b"], dtype=object)
    assert _fingerprint("sha1", ids, ["x"]) != _fingerprint("sha2", ids, ["x"])


def test_fingerprint_detects_class_order_change(cfg) -> None:
    """Reordered classes silently permute every logit column."""
    ids = np.array(["a", "b"], dtype=object)
    assert _fingerprint("sha", ids, ["x", "y"]) != _fingerprint("sha", ids, ["y", "x"])


# --- load_extraction rejection ------------------------------------------------


def test_load_accepts_a_consistent_file(tmp_path, cfg) -> None:
    path = _write(tmp_path, cfg, ["i1", "i2", "i3"])
    ext = load_extraction(cfg, path, ["i1", "i2", "i3"])
    assert len(ext) == 3
    assert ext.taxon_classes == list(cfg.taxon_classes)


def test_load_rejects_a_tampered_fingerprint(tmp_path, cfg) -> None:
    path = _write(tmp_path, cfg, ["i1", "i2"], corrupt=True)
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        load_extraction(cfg, path)


def test_load_rejects_a_stale_class_order(tmp_path, cfg) -> None:
    """A taxonomy edit between extraction and evaluation must not pass."""
    shuffled = list(cfg.taxon_classes)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    path = _write(tmp_path, cfg, ["i1"], taxon_classes=shuffled)
    with pytest.raises(RuntimeError, match="different taxon class order"):
        load_extraction(cfg, path)


def test_load_rejects_manifest_mismatch(tmp_path, cfg) -> None:
    path = _write(tmp_path, cfg, ["i1", "i2"])
    with pytest.raises(RuntimeError, match="does not match the current manifest"):
        load_extraction(cfg, path, ["i1", "i9"])


def test_load_rejects_length_mismatch(tmp_path, cfg) -> None:
    path = _write(tmp_path, cfg, ["i1", "i2"])
    with pytest.raises(RuntimeError, match="does not match the current manifest"):
        load_extraction(cfg, path, ["i1"])
