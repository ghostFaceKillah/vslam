import cv2
import numpy as np
from typing import Tuple

from utils.colors import BGRCuteColors
from utils.custom_types import BGRImageArray, Channels, WidthPx, HeightPx, BGRColor


def magnify(img: BGRImageArray, factor: float = 0.5) -> BGRImageArray:
    """ Convenience wrapper for making image smaller or bigger. """
    return cv2.resize(img, dsize=(0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def take_crop_around(
    img: BGRImageArray,
    around_point: Tuple[int, int],  # h, w
    crop_size: Tuple[int, int],   # h, w
):
    h, w = around_point
    crop_h, crop_w = crop_size
    im_h, im_w = img.shape[:2]

    from_h = max(0, h - crop_h // 2)
    to_h = min(im_h, h + crop_h // 2)

    from_w = max(0, w - crop_w // 2)
    to_w = min(im_w, w + crop_w // 2)

    return np.copy(img[from_h:to_h, from_w:to_w])


def get_canvas(
    shape: Tuple[HeightPx, WidthPx, Channels],
    background_color: BGRColor = BGRCuteColors.DARK_BLUE
):
    return np.ones(shape, dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
