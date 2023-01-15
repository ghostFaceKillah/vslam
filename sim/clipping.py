from typing import List

import attr
import numpy as np

from vslam.math import normalize_vector
from vslam.types import CameraIntrinsics, Vector3d


@attr.define
class ClippingSurfaces:
    right_clipping_surface_normal: Vector3d
    left_clipping_surface_normal: Vector3d
    upper_clipping_surface_normal: Vector3d
    lower_clipping_surface_normal: Vector3d

    @classmethod
    def from_screen_dimensions_and_cam_intrinsics(
        cls,
        screen_h: int,
        screen_w: int,
        cam_intrinsics: CameraIntrinsics
    ) -> 'ClippingSurfaces':
        extreme_left = - cam_intrinsics.cx / cam_intrinsics.fx
        extreme_right = (screen_w - cam_intrinsics.cx) / cam_intrinsics.fx
        extreme_up = - cam_intrinsics.cy / cam_intrinsics.fy
        extreme_down = (screen_h - cam_intrinsics.cy) / cam_intrinsics.fy

        a = normalize_vector(np.array([extreme_right, 0, 1.]))
        b = np.array([0.0, -1.0, 0.0])
        right_clipping_surface_normal = np.cross(b, a)

        a = normalize_vector(np.array([extreme_left, 0, 1.]))
        b = np.array([0.0, 1.0, 0.0])
        left_clipping_surface_normal = np.cross(b, a)

        a = normalize_vector(np.array([0., extreme_up, 1.]))
        b = np.array([-1.0, 0.0, 0.0])
        upper_clipping_surface_normal = np.cross(b, a)

        a = normalize_vector(np.array([0., extreme_down, 1.]))
        b = np.array([1.0, 0.0, 0.0])
        lower_clipping_surface_normal = np.cross(b, a)

        return cls(
            right_clipping_surface_normal=right_clipping_surface_normal,
            left_clipping_surface_normal=left_clipping_surface_normal,
            upper_clipping_surface_normal=upper_clipping_surface_normal,
            lower_clipping_surface_normal=lower_clipping_surface_normal,
        )

    def to_list(self) -> List[Vector3d]:
        return [
            self.right_clipping_surface_normal,
            self.left_clipping_surface_normal,
            self.upper_clipping_surface_normal,
            self.lower_clipping_surface_normal,
        ]