from typing import Tuple, TypeAlias

import attr
import cv2
import numpy as np

from utils.custom_types import BGRImageArray, BGRColor, Array, Pixel


def cv2_circle(
    image: BGRImageArray,
    center_coordinates: Tuple[int, int],   # mind the opencv coordinate flip
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


class CVHersheyFonts:
    FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
    FONT_HERSHEY_PLAIN = cv2.FONT_HERSHEY_PLAIN
    FONT_HERSHEY_DUPLEX = cv2.FONT_HERSHEY_DUPLEX
    FONT_HERSHEY_COMPLEX = cv2.FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_TRIPLEX = cv2.FONT_HERSHEY_TRIPLEX
    FONT_HERSHEY_COMPLEX_SMALL = cv2.FONT_HERSHEY_COMPLEX_SMALL
    FONT_HERSHEY_SCRIPT_SIMPLEX = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_HERSHEY_SCRIPT_COMPLEX = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    FONT_ITALIC = cv2.FONT_ITALIC

CvFontFace: TypeAlias = int


@attr.s(auto_attribs=True)
class CV2GetTextSizeOutputs:
    # https://www.google.com/books/edition/Mastering_OpenCV_4_with_Python/w86PDwAAQBAJ?hl=en&gbpv=1&dq=cv2.getTextSize&pg=PA123&printsec=frontcover
    width_px: int
    height_px: int
    baseline: int   # h coordinate of the baseline relative to the bottom-most text
    # baseline is the line that touches 't' and 'a' on the belly, but is crossed by 'g', 'q' and 'j'


def cv2_get_text_size(
    text: str,
    font_face: CvFontFace,
    font_scale: float,
    font_thickness: int,
) -> CV2GetTextSizeOutputs:
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, font_thickness)
    return CV2GetTextSizeOutputs(
        width_px=text_width,
        height_px=text_height,
        baseline=baseline
    )


def cv2_put_text(
        image: BGRImageArray,
        text: str,
        at: tuple[int, int] = (10, 10),
        font_face: CvFontFace = cv2.FONT_HERSHEY_PLAIN,
        color: BGRColor = (0, 0, 0),
        scale: float = 1.0,
        thickness: int = 1,
):

    cv2.putText(image, text, at, font_face, scale, color, thickness)


def cv2_fill_poly(
        polygon_pts: Array['N,2', np.int32],
        background: BGRImageArray,
        color: BGRColor
):
    cv2.fillPoly(background, [polygon_pts], color=color)
    return background


def cv2_line(
    image: BGRImageArray,
    start_point: Pixel,
    end_point: Pixel,
    color: BGRColor,
    thickness: int
) -> BGRImageArray:
    cv2.line(image, start_point, end_point, color, thickness)
    return image
