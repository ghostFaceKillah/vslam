import cv2
from typing import Tuple

from custom_types import BGRImageArray, BGRColor


def cv2_circle(
    image: BGRImageArray,
    center_coordinates: Tuple[int, int],   # maybe also int works ?
    radius: int,
    color: BGRColor,
    thickness: int = 1,   # -1 if filled
):
    """ thickness: It is the thickness of the circle border line in px. Thickness of -1 px will fill the circle shape by the specified color. """
    cv2.circle(
        image,
        center_coordinates,
        radius,
        color,
        thickness
    )