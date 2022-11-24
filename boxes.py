import attr
import numpy as np

from colors import BGRCuteColors, BGRColor
from custom_types import ImageArray, BGRImageArray
from utils.enum import StrEnum

"""
inputs:
    a bunch of images
    a bunch of Rows, Cols that have panel names
    
    
Packing
    size
    layout
"""

from typing import Tuple, Dict, List, Union, Protocol, runtime_checkable
import attr

import enum

@attr.s(auto_attribs=True)
class PxCoords:
    h: int   # 0 is top
    w: int   # 0 is top


HeightPx = int
WidthPx = int
ImageName = str


@attr.s(auto_attribs=True)
class Packing:
    elem_name_to_px: Dict[ImageName, PxCoords]
    size: Tuple[HeightPx, WidthPx]

    def render(
        self,
        name_to_img: Dict[ImageName, ImageArray],
        background_color: BGRColor = BGRCuteColors.DARK_BLUE
    ):
        canvas = np.ones(self.size + (3,), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

        for elem, px_coords in self.elem_name_to_px.items():
            assert elem in name_to_img, f"Cannot find image {elem} in {name_to_img=}"
            img = name_to_img[elem]
            canvas[px_coords.h:px_coords.h+img.shape[0], px_coords.w:px_coords.w+img.shape[1]] = img
        return canvas


@runtime_checkable
class Packer(Protocol):

    def pack(self, name_to_img: Dict[str, ImageArray]) -> Packing:
        ...

# class PaddedBox():
#     pass

class _Ordering(StrEnum):
    VERTICAL = 'vertical'
    HORIZONTAL = 'horizontal'


def _get_size_of_combined_packings(
        packings: List[Packing],
        ordering: _Ordering
) -> Tuple[HeightPx, WidthPx]:

    height = 0
    width = 0
    for packing in packings:
        h, w = packing.size

        if ordering == _Ordering.HORIZONTAL:
            height, width = max(height, h), width + w
        elif ordering == _Ordering.VERTICAL:
            height, width = height + h, max(width, w)
        else:
            raise ValueError('Unknown ordering')

    return height, width


@attr.s(auto_attribs=True)
class _RowCol(Packer):
    _items: List[Union[Packer, ImageName]]
    _ordering: _Ordering

    def _items_to_packings(self, name_to_img: Dict[str, ImageArray]) -> List[Packing]:
        packings = []

        # render items into packings
        for item in self._items:
            if isinstance(item, Packer):
                packing = item.pack(name_to_img)
            else:
                assert isinstance(item, str)
                packing = Packing(
                    elem_name_to_px={item: PxCoords(h=0, w=0)},
                    size=name_to_img[item].shape[:2]
                )
            packings.append(packing)

        return packings

    def pack(self, name_to_img: Dict[str, ImageArray]) -> Packing:

        packings = self._items_to_packings(name_to_img)
        height, width = _get_size_of_combined_packings(packings, self._ordering)

        elem_name_to_px: Dict[ImageName, PxCoords] = {}
        h_offset, w_offset = 0, 0

        for packing in packings:
            pack_h, pack_w = packing.size

            if self._ordering == _Ordering.VERTICAL:
                packing_h_offset = h_offset
                packing_w_offset = (width - pack_w) // 2
                h_offset += pack_h
            elif self._ordering == _Ordering.HORIZONTAL:
                packing_h_offset = (height - pack_h) // 2
                packing_w_offset = w_offset
                w_offset += pack_w
            else:
                raise ValueError('Unknown ordering')

            for name, px_coords in packing.elem_name_to_px.items():
                elem_name_to_px[name] = PxCoords(h=px_coords.h + packing_h_offset, w=px_coords.w + packing_w_offset)

        return Packing(
            elem_name_to_px=elem_name_to_px,
            size=(height, width)
        )

    def render(self, name_to_img: Dict[str, ImageArray]) -> BGRImageArray:
        return self.pack(name_to_img).render(name_to_img)


class Col(_RowCol):
    def __init__(self, *args):
        _RowCol.__init__(self, items=list(args), ordering=_Ordering.VERTICAL)


class Row(_RowCol):
    def __init__(self, *args):
        _RowCol.__init__(self, items=list(args), ordering=_Ordering.HORIZONTAL)


def test_basic_row_and_col_behaviour():

    inputs = {
        "a": np.zeros(shape=(640, 480), dtype=np.uint8),
        "b": np.zeros(shape=(1024, 768, 3), dtype=np.uint8),
        "c": np.zeros(shape=(2880, 1440), dtype=np.uint8)
    }

    assert Row("a", "b", "c").pack(inputs).size == (2880, 2688)
    assert Col("a", "b", "c").pack(inputs).size == (4544, 1440)
    assert Col(Row("a"), Row("b"), Row("c")).pack(inputs).size == (4544, 1440)
    assert Row(Col("a"), Col("b"), Col("c")).pack(inputs).size == (2880, 2688)
    assert Row(Col("a", Row("b")), "c").pack(inputs).size == (2880, 2208)


def test_integration():

    inputs = {
        "a": np.array(BGRCuteColors.CYAN, dtype=np.uint8) * np.ones(shape=(640, 480, 3), dtype=np.uint8),
        "b": np.array(BGRCuteColors.SALMON, dtype=np.uint8) * np.ones(shape=(100, 600, 3), dtype=np.uint8),
        "c": np.array(BGRCuteColors.CRIMSON, dtype=np.uint8) * np.ones(shape=(288, 244, 3), dtype=np.uint8),
        "d": np.array(BGRCuteColors.SUN_YELLOW, dtype=np.uint8) * np.ones(shape=(288, 244, 3), dtype=np.uint8)
    }
    Col(Row("a", "b"), Row("c", "d")).render(inputs)
    Col("a", "b", "c", "d").render(inputs)


if __name__ == '__main__':
    test_basic_row_and_col_behaviour()
    test_integration()
