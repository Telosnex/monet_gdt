"""Pure helpers in lib/, no data needed."""

import numpy as np
from color_space import hex2rgb, rgb2lab
from mood_words import WORDS, wall_percentiles


def test_hex2rgb():
    np.testing.assert_allclose(hex2rgb('#ff8000'), [1.0, 128 / 255, 0.0])


def test_rgb2lab_reference_values():
    # The sRGB matrix is rounded to 4 digits, so white lands ~0.01 off neutral.
    np.testing.assert_allclose(rgb2lab([1, 1, 1]), [100, 0, 0], atol=0.02)
    np.testing.assert_allclose(rgb2lab([0, 0, 0]), [0, 0, 0], atol=0.02)
    np.testing.assert_allclose(rgb2lab([1, 0, 0]), [53.24, 80.09, 67.20], atol=0.05)


def test_rgb2lab_batched_shape():
    assert rgb2lab(np.zeros((4, 5, 3))).shape == (4, 5, 3)


def test_wall_percentiles_mid_rank():
    pop = {w: [0.0, 1.0, 2.0] for w in WORDS}
    raw = {w: 1.0 for w in WORDS}
    np.testing.assert_allclose(wall_percentiles(raw, pop), 0.5)
    lo = wall_percentiles({w: -9.0 for w in WORDS}, pop)
    hi = wall_percentiles({w: 9.0 for w in WORDS}, pop)
    assert (lo > 0).all() and (hi < 1).all() and (lo < hi).all()
