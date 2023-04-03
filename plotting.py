from typing import Tuple, Dict, List, Union, Protocol, runtime_checkable, Generator

import attr
import cv2

from utils.colors import BGRCuteColors, BGRColor
from utils.custom_types import ImageArray, BGRImageArray, HeightPx, WidthPx
from utils.cv2_but_its_typed import cv2_get_text_size
from utils.enum_utils import StrEnum
from utils.image import get_canvas


@attr.s(auto_attribs=True)
class PxCoords:
    h: int   # 0 is top
    w: int   # 0 is top


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
        canvas = get_canvas(self.size + (3,), background_color)

        for elem, px_coords in self.elem_name_to_px.items():
            assert elem in name_to_img, f"Cannot find image {elem} in {name_to_img=}"
            img = name_to_img[elem]
            canvas[px_coords.h:px_coords.h+img.shape[0], px_coords.w:px_coords.w+img.shape[1]] = img
        return canvas


@runtime_checkable
class Packer(Protocol):
    # TODO: Rename to ImagePacker

    def pack(self, name_to_img: Dict[str, ImageArray]) -> Packing:
        ...

    def render(self, name_to_img: Dict[str, ImageArray]) -> ImageArray:
        ...



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


def _items_and_images_to_packings(
        items: List[Union[Packer, ImageName]],
        name_to_img: Dict[str, ImageArray]
    ):
        packings = []

        # render items into packings
        for item in items:
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


@attr.s(auto_attribs=True)
class Padding(Packer):
    _item: Union[Packer, ImageName]
    _padding: int = 20

    def pack(self, name_to_img: Dict[str, ImageArray]) -> Packing:
        assert len(name_to_img)

        packing = _items_and_images_to_packings([self._item], name_to_img)[0]
        height, width = packing.size

        height += 2 * self._padding
        width += 2 * self._padding

        elem_name_to_px: Dict[ImageName, PxCoords] = {}
        for name, px_coords in packing.elem_name_to_px.items():
            elem_name_to_px[name] = PxCoords(h=px_coords.h + self._padding, w=px_coords.w + self._padding)

        return Packing(elem_name_to_px=elem_name_to_px, size=(height, width))


@attr.s(auto_attribs=True)
class _RowCol(Packer):
    _items: List[Union[Packer, ImageName]]
    _ordering: _Ordering

    def pack(self, name_to_img: Dict[str, ImageArray]) -> Packing:

        packings = _items_and_images_to_packings(self._items, name_to_img)
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

        return Packing(elem_name_to_px=elem_name_to_px, size=(height, width))

    def render(self, name_to_img: Dict[str, ImageArray]) -> BGRImageArray:
        return self.pack(name_to_img).render(name_to_img)


class Col(_RowCol):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Generator):
            items = list(args[0])
        else:
            items = list(args)
        _RowCol.__init__(self, items=items, ordering=_Ordering.VERTICAL)



class Row(_RowCol):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Generator):
            items = list(args[0])
        else:
            items = list(args)
        _RowCol.__init__(self, items=items, ordering=_Ordering.HORIZONTAL)


@attr.s(auto_attribs=True)
class FontStyle:
    color: BGRColor = BGRCuteColors.OFF_WHITE
    thickness: int = 1
    face: int = cv2.FONT_HERSHEY_PLAIN
    scale: float = 1


@attr.s(auto_attribs=True)
class TextRenderer:
    font_style: FontStyle = attr.Factory(FontStyle)
    vspace: float = 0.4
    padding: int = 15
    background_color: BGRColor = BGRCuteColors.DARK_BLUE

    def render(self, txt: str) -> BGRImageArray:
        lines = txt.split('\n')

        longest_line = max(lines, key=len)

        text_size = cv2_get_text_size(longest_line, self.font_style.face, self.font_style.scale, self.font_style.thickness)
        width = text_size.width_px + self.padding
        height = int(len(lines) * text_size.height_px * (1 + self.vspace))
        canvas = get_canvas((height, width, 3), self.background_color)

        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (0, int(text_size.baseline* 2 + i * (1 + self.vspace) * text_size.height_px)),
                fontFace=self.font_style.face,
                fontScale=self.font_style.scale,
                color=self.font_style.color,
                thickness=self.font_style.thickness,
                bottomLeftOrigin=False
            )

        return canvas


if __name__ == '__main__':
    img = TextRenderer().render('heheh \npoetry of the heart \n okekoekeoke')
    import cv2
    cv2.imshow('hehe', img)
    cv2.waitKey(-1)
