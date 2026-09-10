"""The empty-feeder miner, and the two failures that shaped it.

An "empty feeder" frame containing a bird is the worst label in this corpus: it
teaches the model that the thing it exists to detect is the thing it should
ignore. Everything here guards a mistake that was actually made and measured on
the 2026-09 footage, not a hypothetical one.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdcam.data.mine_negatives import (
    _window,
    contrast_is_usable,
    feeder_port,
    port_box,
    propose_threshold,
)


def _scene(w: int = 455, h: int = 256, port_x: int = 240, port_y: int = 150) -> np.ndarray:
    """A grey scene with one red feeder port in it."""
    a = np.full((h, w, 3), 120.0, dtype=np.float32)
    a[port_y - 8 : port_y + 8, port_x - 12 : port_x + 12] = (210, 30, 40)
    return a


# --- finding the port ---------------------------------------------------------


def test_finds_the_red_port() -> None:
    cx, cy = feeder_port(_scene())
    assert abs(cx - 240) <= 4 and abs(cy - 150) <= 4


def test_no_port_is_an_error_not_a_guess() -> None:
    """A wrong port location silently moves the search window off the bird."""
    with pytest.raises(ValueError):
        feeder_port(np.full((256, 455, 3), 120.0, dtype=np.float32))


# --- the window bug -----------------------------------------------------------
#
# This is the one that let a Cape White-eye through. The port box was CLIPPED to
# the frame, so when the port sat near an edge the port ended up at the edge of
# its own box and a bird on the far side of it fell outside entirely. It scored
# as empty and it looked empty in review, because neither was looking at it.


def test_window_keeps_full_width_at_the_left_edge() -> None:
    lo, hi = _window(centre=5, half=50, limit=455)
    assert hi - lo == 100, "window was clipped instead of shifted"
    assert lo >= 0


def test_window_keeps_full_width_at_the_right_edge() -> None:
    lo, hi = _window(centre=450, half=50, limit=455)
    assert hi - lo == 100, "window was clipped instead of shifted"
    assert hi <= 455


def test_window_contains_the_centre_even_when_shifted() -> None:
    for c in (0, 5, 227, 450, 455):
        lo, hi = _window(c, 50, 455)
        assert lo <= min(c, 454) < hi


def test_port_box_lies_inside_the_frame_wherever_the_port_is() -> None:
    for px in (20, 240, 440):
        x0, y0, x1, y1 = port_box(_scene(port_x=px))
        assert 0 <= x0 < x1 <= 455
        assert 0 <= y0 < y1 <= 256
        assert x0 <= px < x1, "the port fell outside its own box"


# --- the session gate ---------------------------------------------------------


def test_compressed_score_range_is_rejected() -> None:
    """Wind on the tarp swamps the bird signal; 20260825_171725 measured 2.40."""
    ok, dyn = contrast_is_usable(np.random.default_rng(0).normal(40, 2, 500))
    assert not ok and dyn < 2.7


def test_wide_score_range_is_accepted() -> None:
    ok, dyn = contrast_is_usable(np.random.default_rng(0).uniform(5, 60, 500))
    assert ok and dyn >= 2.7


# --- the threshold ------------------------------------------------------------


def test_threshold_is_a_percentile_not_a_split() -> None:
    """Otsu was tried and rejected: it assumes two modes, and this is one skewed one."""
    s = np.concatenate([np.linspace(20, 35, 850), np.linspace(35, 90, 150)])
    thr = propose_threshold(s, 0.15)
    assert (s < thr).mean() == pytest.approx(0.15, abs=0.02)
