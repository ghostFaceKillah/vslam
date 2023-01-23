import jax.numpy as np

from sim.render_utils import get_line_pixels


def test_short_line():
    from_px = 2, 3
    to_px = 3, 2

    expected = np.array([[3, 2], [2, 3]], dtype=np.int32)

    assert np.array_equal(get_line_pixels(from_px, to_px), expected)
    assert np.array_equal(get_line_pixels(to_px, from_px), expected)


def test_longer_line():
    from_px = 0, 0
    to_px = 4, 4

    expected = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32)
    assert np.array_equal(get_line_pixels(from_px, to_px), expected)
    assert np.array_equal(get_line_pixels(to_px, from_px), expected)


def test_almost_horizontal_line():
    from_px = 2, 1
    to_px = 3, 100

    line_px = get_line_pixels(from_px, to_px)
    assert len(line_px) == 100
    assert get_line_pixels(from_px, to_px) == get_line_pixels(to_px, from_px)


def test_horizontal_line():
    from_px = 2, 1
    to_px = 2, 5

    expected = np.array([[2, 1], [2, 2], [2, 3], [2, 4], [2, 5]], dtype=np.int32)
    assert np.array_equal(get_line_pixels(from_px, to_px), expected)
    assert np.array_equal(get_line_pixels(to_px, from_px), expected)


def test_vertical_line():
    from_px = 1, 2
    to_px = 5, 2

    expected = np.array([[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]], dtype=np.int32)
    assert np.array_equal(get_line_pixels(from_px, to_px), expected)
    assert np.array_equal(get_line_pixels(to_px, from_px), expected)
