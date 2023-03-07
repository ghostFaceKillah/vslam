from typing import List

import attr
from jax import numpy as np

from sim.clipping import ClippingSurfaces
from sim.ui import InteractionTransforms
from utils.colors import BGRCuteColors
from utils.custom_types import Array, BGRColor, BGRImageArray
from vslam.poses import get_SE3_pose
from vslam.types import TransformSE3, CameraIntrinsics, CameraPoseSE3


@attr.define
class RenderTriangle3d:
    """ a triangle floating in space """
    points: Array['3,4', np.float64]   # three 3d points
    front_face_color: BGRColor = BGRCuteColors.GRASS_GREEN
    back_face_color: BGRColor = BGRCuteColors.CRIMSON

    def mutate(self, transform: TransformSE3) -> 'RenderTriangle3d':
        result = transform @ self.points.T
        return RenderTriangle3d(
            points=result.T,
            front_face_color=self.front_face_color,
            back_face_color=self.back_face_color,
        )


RenderTrianglesPointsInCam = Array['N,3,4', np.float64]


@attr.define
class CameraSpecs:
    """ Mix of intrinsics and extrinsics """
    screen_h: int
    screen_w: int
    distance_between_eyes: float
    cam_intrinsics: CameraIntrinsics
    clipping_surfaces: ClippingSurfaces

    @classmethod
    def from_default(
        cls,
        screen_h: int = 480,
        screen_w: int = 640,
        f_mod: float = 2.0,   # higher f_mod -> less distortion, less field of view
    ):
        cam_intrinsics = CameraIntrinsics(
            fx=screen_w / 4 * f_mod,
            fy=screen_h / 3 * f_mod,
            cx=screen_w / 2,
            cy=screen_h / 2,
        )
        return cls(
            screen_h=screen_h,
            screen_w=screen_w,
            distance_between_eyes=2.0,
            cam_intrinsics=cam_intrinsics,
            clipping_surfaces=ClippingSurfaces.from_screen_dimensions_and_cam_intrinsics(screen_h, screen_w, cam_intrinsics),
        )

    def get_pose_of_right_cam_in_left_cam(self) -> CameraPoseSE3:
        return get_SE3_pose(y=self.distance_between_eyes)

    def get_pose_of_left_cam_in_baselink(self) -> CameraPoseSE3:
        return get_SE3_pose(y=-self.distance_between_eyes / 2)


@attr.define
class Observation:
    left_eye_img: BGRImageArray
    right_eye_img: BGRImageArray
    bev_img: BGRImageArray   # birdseye view image
    baselink_pose: CameraPoseSE3   # left eye to the left of this, right eye to the right of this
    frame_idx: int
    timestamp: float   # in seconds since epoch, as per python convention


@attr.define
class Action:
    transforms: InteractionTransforms
    end: bool

    @classmethod
    def empty(cls):
        return cls(transforms=InteractionTransforms.empty(), end=False)

    @classmethod
    def done(cls):
        return cls(transforms=InteractionTransforms.empty(), end=True)


@attr.define
class Recording:
    camera_specs: CameraSpecs
    observations: List[Observation] = attr.ib(factory=list)

    def record_observation(self, obs: Observation):
        self.observations.append(obs)