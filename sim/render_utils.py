"""

- Render a triangle on canvas

- Render bird's eye view triangle.
- Render view port

"""

from typing import List

import attr
import jax.numpy as np

from utils.custom_types import Array, Pixel, PixelCoordArray, MaskArray


@attr.define
class TriangleScanlines:
    min_y: int
    max_y: int
    min_xs: List[int]   # maps from given y to min_x
    max_xs: List[int]


def _get_triangle_scanline_definitions(
        triangle_px_coord: PixelCoordArray
) -> TriangleScanlines:
    """ Get coordinates of all pixels in a given triangle.
     Similar to cv2.polyfill, but implemented here (TM).
    Based on https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html """
    assert triangle_px_coord.shape == (3, 2)

    # naming convention in this function: p = x, y
    p0 = triangle_px_coord[0, 0], triangle_px_coord[0, 1]
    p1 = triangle_px_coord[1, 0], triangle_px_coord[1, 1]
    p2 = triangle_px_coord[2, 0], triangle_px_coord[2, 1]

    # sort points such that y0 < y1 < y2
    if p1[1] < p0[1]:
        p1, p0 = p0, p1
    if p2[1] < p0[1]:
        p2, p0 = p0, p2
    if p2[1] < p1[1]:
        p2, p1 = p1, p2

    short_line_1 = get_line_pixels(p0, p1)
    short_line_2 = get_line_pixels(p1, p2)
    short_line = np.concatenate([short_line_1[:-1], short_line_2[:-1]])

    long_line = get_line_pixels(p0, p2)

    max_y = p2[1]
    min_y = p0[1]

    min_xs = [np.inf for _ in range(max_y - min_y + 1)]
    max_xs = [-np.inf for _ in range(max_y - min_y + 1)]

    for x, y in short_line:
        min_xs[y - min_y] = min(int(x), min_xs[y - min_y])
        max_xs[y - min_y] = max(int(x), max_xs[y - min_y])

    for x, y in long_line:
        min_xs[y - min_y] = min(int(x), min_xs[y - min_y])
        max_xs[y - min_y] = max(int(x), max_xs[y - min_y])

    return TriangleScanlines(min_y=min_y, max_y=max_y, min_xs=min_xs, max_xs=max_xs)


def get_triangle_pixel_indices(triangle_px_coord: PixelCoordArray) -> PixelCoordArray:
    """ Get coordinates of all pixels in a given triangle.
    Based on https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html """

    scanlines = _get_triangle_scanline_definitions(triangle_px_coord)

    for i in range(scanlines.max_y - scanlines.min_y + 1):
        y = i + scanlines.min_y
        min_x = scanlines.min_xs[i]
        max_x = scanlines.max_xs[i]

        scanline = np.array([[x, y] for x in range(min_x, max_x + 1)], dtype=np.int32)
        scanlines.append(scanline)

    return np.concatenate(scanlines)


def get_screen_pixel_coords(screen_h: int, screen_w: int) -> Array['H,W,2', np.int32]:
    ws = np.arange(0, screen_w, dtype=np.int32)
    hs = np.arange(0, screen_h, dtype=np.int32)
    return np.meshgrid(ws, hs)


def _get_triangle_mask_jaxed(
        screen_h: int,
        screen_w: int,
):
    """ I think it should be possible to jaxify the whole thing by getting rid of the scanlines object
    """
    pass


def get_triangle_mask(
        screen_h: int,
        screen_w: int,
        triangle_px_coord: PixelCoordArray
) -> MaskArray:
    """ Similar to cv2.polyfill, but implemented here (TM).
    Based on https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html """

    scanlines = _get_triangle_scanline_definitions(triangle_px_coord)
    xs, ys = get_screen_pixel_coords(screen_h, screen_w)

    pre_array_len = scanlines.min_y
    post_array_len = int(screen_h - scanlines.min_y - len(scanlines.min_xs))
    inf = screen_w + 1
    minf = -1

    well_indexed_max_xs = np.concatenate([
        minf * np.ones(pre_array_len, dtype=np.int32),
        np.array(scanlines.max_xs),
        minf * np.ones(post_array_len, dtype=np.int32),
    ])

    well_indexed_min_xs = np.concatenate([
        inf * np.ones(pre_array_len, dtype=np.int32),
        np.array(scanlines.min_xs),
        inf * np.ones(post_array_len, dtype=np.int32),
    ])

    mask = (xs <= well_indexed_max_xs[:, np.newaxis]) & (xs >= well_indexed_min_xs[:, np.newaxis])
    x = 1


    pre_array_len = scanlines.min_y

    # iys >= scanlines.min_y
    # iys <= scanlines.max_y

    min_xs = np.where()
    np.arange()



    x = 1




def _interpolate_worker(from_px: Pixel, to_px: Pixel) -> PixelCoordArray:
    """
    Interpolate between `from_px` and `to_px`, but we can assume that the line is more 'horizontal'
    then 'vertical'. In this way we know that our interpolation will be "dense enough" along `y` axis.
    and we won't have discontinuities.

    end inclusive!
    """

    assert to_px[0] != from_px[0], "Edge case of vertical lines should be handled elsewhere"

    if to_px[0] < from_px[0]:
        # we want to draw lines from left to right in this function
        from_px, to_px = to_px, from_px

    i0, d0 = from_px
    i1, d1 = to_px

    assert i1 > i0, "By usage convention, we want to draw lines from left to right in this function"
    assert i1-i0 >= d1-d0, "There will be discontinuities if used this way"

    slope = (d1 - d0) / (i1 - i0)
    ds = (np.arange(i1 - i0 + 1) * slope + d0).astype(np.int32)
    is_ = (np.arange(i1 - i0 + 1) + i0).astype(np.int32)

    return np.stack([is_, ds], axis=1)


def get_line_pixels(from_px: Pixel, to_px: Pixel) -> Array['K,2', np.int32]:
    """
    Interpolate between two given pixels. Return indices of pixels that lie inbetween two given pixels

    Haha, I am following this website here,
    https://gabrielgambetta.com/computer-graphics-from-scratch/06-lines.html
    but often, especially for the sake of computing pixel indices, that's totally not the way to do it.
    Here's the way to do it "properly" https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """

    if from_px == to_px:
        # it's a very short line !
        return np.array([from_px], dtype=np.int32)

    if abs(to_px[0] - from_px[0]) > abs(to_px[1] - from_px[1]):
        # The line is more or less horizontal
        return _interpolate_worker(from_px, to_px)
    else:
        # the line is more vertical
        return _interpolate_worker(from_px[::-1], to_px[::-1])[:, ::-1]




if __name__ == '__main__':
    triangle_px = np.array([[1, 1], [1, 5], [5, 1]], dtype=np.int32)
    get_triangle_mask(screen_w=640, screen_h=480, triangle_px_coord=triangle_px)

