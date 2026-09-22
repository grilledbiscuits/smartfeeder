"""Pillarbox detection: portrait video padded into a landscape frame.

The failure this guards against is silent -- barred frames train and score
without complaint, and only show up as a class the model confuses with hands.
"""

import numpy as np
import pytest

from birdcam.data.letterbox import bar_columns, is_letterboxed


def barred(content=120.0, w=455, h=256, left=155, right=300):
    a = np.zeros((h, w, 3), dtype=np.float32)
    a[:, left:right] = content
    a[:, left + 10 : left + 30, 0] = 200  # some structure in the picture
    return a


def test_finds_the_bars():
    assert bar_columns(barred()) == (155, 300)


def test_plain_frame_is_not_letterboxed():
    assert not is_letterboxed(np.full((256, 455, 3), 120.0, dtype=np.float32))


def test_portrait_frame_is_not_letterboxed():
    """455x256 the other way up is real portrait footage, not padding."""
    assert not is_letterboxed(np.full((455, 256, 3), 120.0, dtype=np.float32))


def test_night_frame_is_not_letterboxed():
    """A dark scene is dark everywhere; a pillarbox is dark only at the edges."""
    assert not is_letterboxed(np.full((256, 455, 3), 4.0, dtype=np.float32))


def test_dark_content_between_bars_is_not_claimed():
    assert not is_letterboxed(barred(content=8.0))


def test_one_bright_patch_in_a_night_frame_is_not_bars():
    a = np.full((256, 455, 3), 4.0, dtype=np.float32)
    a[:, 220:240] = 180.0
    assert not is_letterboxed(a)


def test_bars_must_reach_both_edges():
    a = barred()
    a[:, :20] = 120.0  # bright strip outside the "bar"
    assert not is_letterboxed(a)


@pytest.mark.parametrize("left,right", [(155, 300), (100, 355), (180, 320)])
def test_crop_recovers_the_picture(left, right):
    a = barred(left=left, right=right)
    lo, hi = bar_columns(a)
    assert (lo, hi) == (left, right)
    assert a[:, lo:hi].mean() > 0
