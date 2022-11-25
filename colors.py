from typing import Tuple

from custom_types import BGRColor


class BGRCuteColors:
    DARK_GRAY = 20, 20, 20

    PURPLE = 119, 17, 136
    CRIMSON = 85, 51, 170
    SALMON = 102, 102, 204
    ORANGE = 68, 153, 238

    SUN_YELLOW = 0, 221, 238
    GRASS_GREEN = 85, 221, 153
    CYAN = 136, 221, 68
    TURQUOISE = 187, 204, 34

    OCEAN_BLUE = 204, 187, 0
    SKY_BLUE = 204, 153, 0
    DARK_BLUE = 187, 102, 51
    VIOLET = 153, 51, 102


def _convert_12_bit_to_bgr(clr_12_bit: str) -> BGRColor:
    r, g, b = [int(2 * b, 16) for b in clr_12_bit]
    return b, g, r


def _steal_some_colors():
    # stealin from https://iamkate.com/data/12-bit-rainbow/
    colors = [
        ('817', 'purple'),
        ('a35', 'crimson'),
        ('c66', 'salmon'),
        ('e94', 'orange'),
        ('ed0', 'sun_yellow'),
        ('9d5', 'grass_green'),
        ('4d8', 'cyan'),
        ('2cb', 'turquoise'),
        ('0bc', 'ocean_blue'),
        ('09c', 'sky_blue'),
        ('36b', 'dark_blue'),
        ('639', 'violet'),
    ]
    for clr_12_bit, name in colors:
        b, g, r = _convert_12_bit_to_bgr(clr_12_bit)
        print(f"{name.upper()} = {b}, {g}, {r}")
