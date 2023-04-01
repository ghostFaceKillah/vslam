import attr
import numpy as np

from vslam.types import CameraIntrinsicMatrix


@attr.s(auto_attribs=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    screen_h: int   # traditionally not contained in intrinsics
    screen_w: int

    def to_matrix(self) -> CameraIntrinsicMatrix:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_homo_cam_coords_to_px_coords_matrix(self):
        # CamCoords3dHomog to PxCoords2d matrix
        return np.array([
            [0, self.fy, self.cy],
            [self.fx, 0, self.cx],
        ], dtype=np.float64)
