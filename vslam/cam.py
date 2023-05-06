import attr


@attr.s(auto_attribs=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    screen_h: int   # traditionally not contained in intrinsics
    screen_w: int
