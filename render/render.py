"""
Small demo of naive rendering
"""
from typing import Optional

import attr
import cv2
import numpy as np

from utils.colors import BGRCuteColors
from utils.custom_types import Array, BGRImageArray
from utils.image import get_canvas
from vslam.math import normalize_vector, dot_product
from vslam.poses import get_SE3_pose
from vslam.transforms import get_world_to_cam_coord_flip_matrix, SE3_inverse, the_cv_flip
from vslam.types import CameraPoseSE3, CameraIntrinsics, Vector3dHomogenous, Vector3d, TransformSE3


# https://github.com/krishauser/Klampt/blob/master/Python/klampt/math/se3.py
# cool reference

@attr.define
class Triangle3d:
    """ a triangle floating in space """
    points: Array['3,4', np.float64]   # three 3d points

    def get_surface_normal(self) -> Vector3dHomogenous:
        # TODO: There has to be a convention around those in rendering community
        vec_1 = (self.points[1] - self.points[0])[:3]
        vec_2 = (self.points[2] - self.points[0])[:3]
        resu = np.cross(vec_1, vec_2)
        return resu / np.linalg.norm(resu)

    def mutate(self, transform: TransformSE3) -> 'Triangle3d':
        result = transform @ self.points.T
        return Triangle3d(points=result.T)


def toy_render_one_triangle(
    screen: BGRImageArray,
    camera_pose: CameraPoseSE3,
    triangle: Triangle3d,
    cam_intrinsics: CameraIntrinsics,
    light_direction: Vector3d
):
    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()

    # we have a triangle in space

    # we bring points to cam coordinate frame
    # 1) camera pose transform
    # 2) coordinate flip
    # cam_coordinates * points_in_cam = world_coordinates * points_in_world
    # points_in_cam = world_to_cam_matrix() * inverse_of_cam_pose * points_in_world
    cam_pose_inv = SE3_inverse(camera_pose)
    cam_points_T = world_to_cam_flip @ cam_pose_inv @ triangle.points.T
    cam_points = cam_points_T.T

    # drop_homogenous coord one
    cam_points = cam_points[:, :3]

    # drop things that are not visible
    cam_points = cam_points[cam_points[:, 2] > 1]

    # project to unit image plane
    projected_to_unit_plane = cam_points / cam_points[:, 2][:, np.newaxis]

    # map to pixel coordinates
    px_coords = (cam_intrinsics.get_homo_cam_coords_to_px_coords_matrix()  @ projected_to_unit_plane.T).T

    # render the triangle if at least one point lies within the image plane
    ...

    # we will want to compute surface normal by computing crossproduct
    normal = triangle.get_surface_normal()

    # we will want to compute color by inner_product(light direction, surface normal)
    dot_product(triangle.get_surface_normal(), light_direction)
    # TODO
    color = BGRCuteColors.GRASS_GREEN

    poly_coords = the_cv_flip(px_coords.round().astype(np.int32))

    cv2.fillPoly(screen, [poly_coords], color)


@attr.s(auto_attribs=True)
class MaybeTransforms:
    camera: Optional[TransformSE3] = None
    scene: Optional[TransformSE3] = None

    @classmethod
    def empty(cls):
        return cls()


def _key_to_maybe_transforms(cv_key: int) -> MaybeTransforms:
    # 0 up, 1 down, 3 right
    if key == -1:
        return MaybeTransforms.empty()
    elif key == 0:   # up
        return MaybeTransforms(scene=get_SE3_pose(pitch=np.deg2rad(10)))
    elif key == 1:   # down
        return MaybeTransforms(scene=get_SE3_pose(pitch=np.deg2rad(-10)))
    elif key == 2:   # left
        return MaybeTransforms(scene=get_SE3_pose(roll=np.deg2rad(10)))
    elif key == 3:   # right
        return MaybeTransforms(scene=get_SE3_pose(roll=np.deg2rad(-10)))
    elif key == ord('w'):
        return MaybeTransforms(camera=get_SE3_pose(x=0.1))
    elif key == ord('s'):
        return MaybeTransforms(camera=get_SE3_pose(x=-0.1))
    elif key == ord('a'):
        return MaybeTransforms(camera=get_SE3_pose(y=-0.1))
    elif key == ord('d'):
        return MaybeTransforms(camera=get_SE3_pose(y=0.1))
    else:
        print(f"Unknown keypress {cv_key} {chr(cv_key)}")
        return MaybeTransforms.empty()


if __name__ == '__main__':
    # I want image to be 640 by 480
    # I want it to map from 4 by 3 meters
    cam_intrinsics = CameraIntrinsics(fx=640 / 4, fy=480 / 3, cx=640 / 2, cy=480 / 2)
    light_direction = normalize_vector(np.array([1.0, -1.0, -8.0]))

    # looking toward +x direction in world frame, +z in camera
    camera_pose: CameraPoseSE3 = get_SE3_pose()

    triangle = Triangle3d(np.array([
        [1.5, -1.0, -1.0, 1.0],
        [1.5,  2.0, -1.0, 1.0],
        [1.5, -1.0, 2.0, 1.0],
    ], dtype=np.float64))

    while True:
        screen = get_canvas(shape=(480, 640, 3), background_color=BGRCuteColors.DARK_GRAY)

        toy_render_one_triangle(screen, camera_pose, triangle, cam_intrinsics, light_direction)

        cv2.imshow('scene', screen)
        key = cv2.waitKey(-1)

        transforms = _key_to_maybe_transforms(key)

        if transforms.scene is not None:
            triangle = triangle.mutate(transforms.scene)

        if transforms.camera is not None:
            camera_pose = transforms.camera @ camera_pose



