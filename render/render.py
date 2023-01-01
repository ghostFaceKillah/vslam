"""
Small demo of naive rendering


"""
import itertools
from typing import Optional, List, Tuple

import attr
import cv2
import numpy as np

from utils.colors import BGRCuteColors
from utils.custom_types import Array, BGRImageArray, BGRColor
from utils.image import get_canvas
from vslam.math import normalize_vector
from vslam.poses import get_SE3_pose
from vslam.transforms import get_world_to_cam_coord_flip_matrix, SE3_inverse, the_cv_flip, homogenize
from vslam.types import CameraPoseSE3, CameraIntrinsics, Vector3d, TransformSE3, Point2d, CamCoords3d, Points2d


# https://github.com/krishauser/Klampt/blob/master/Python/klampt/math/se3.py
# cool reference


class Triangle2d:
    points: Array['3,2', np.float64]

    def to_barycentric(self, pt: Point2d):
        # silly! way better to do it for many points at once
        x, y = pt
        x_1, y_1 = self.points[0, :]
        x_2, y_2 = self.points[1, :]
        x_3, y_3 = self.points[2, :]

        denom = (y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3)

        lambda_1_num = (y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)
        lambda_2_num = (y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)

        lambda_1 = lambda_1_num / denom
        lambda_2 = lambda_2_num / denom
        lambda_3 = 1 - lambda_1 - lambda_2
        return lambda_1, lambda_2, lambda_3

    def to_barycentric_many(self, x: Points2d):
        x_1, y_1 = self.points[0, :]
        x_2, y_2 = self.points[1, :]
        x_3, y_3 = self.points[2, :]
        r_3 = self.points[2, :]

        T = np.array([
            [x_1 - x_3, x_2 - x_3],
            [y_1 - y_3, y_2 - y_3]
        ], np.float64)

        T_inv = np.linalg.inv(T)
        barycentric_partial = T_inv @ (x - r_3)
        barycentric = np.dstack([barycentric_partial, 1 - barycentric_partial.sum(axis=2)])

        return barycentric


def generate_cube_sides() -> List[Tuple[Vector3d, Vector3d, Vector3d]]:
    def _is_nice_triangle(a, b, c):
        diff_ab = b - a
        diff_bc = b - c
        diff_ac = a - c

        m1, m2, _ = sorted([np.abs(diff).sum() / 2 for diff in [diff_ab, diff_bc, diff_ac]])
        _, _, m3 = sorted([np.abs(diff.sum()) // 2 for diff in [diff_ab, diff_bc, diff_ac]])
        return m1 < 1.1 and m2 < 1.1 and m3 == 1

    vals = [-1, 1]
    triplets = [np.array([x, y, z]) for x in vals for y in vals for z in vals]

    sides = [
        np.array(triplet, dtype=np.float64)
        for triplet in itertools.combinations(triplets, 3)
        if _is_nice_triangle(*triplet)
    ]

    # need to renormalize

    return sides

@attr.define
class Triangle3d:
    """ a triangle floating in space """
    points: Array['3,4', np.float64]   # three 3d points

    def get_surface_normal(self) -> Vector3d:
        # TODO: There has to be a convention around those in rendering community
        vec_1 = (self.points[1] - self.points[0])[:3]
        vec_2 = (self.points[2] - self.points[0])[:3]
        resu = np.cross(vec_1, vec_2)
        return resu[:3] / np.linalg.norm(resu)

    def mutate(self, transform: TransformSE3) -> 'Triangle3d':
        result = transform @ self.points.T
        return Triangle3d(points=result.T)


def compute_triangle_sorting_index(
    camera_pose: CameraPoseSE3,
    triangles: List[Triangle3d],
) -> List[int]:

    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()
    cam_pose_inv = SE3_inverse(camera_pose)

    mean_depths = []
    for triangle in triangles:
        cam_points_T = world_to_cam_flip @ cam_pose_inv @ triangle.points.T
        cam_points = cam_points_T.T

        # drop_homogenous coord one
        cam_points = cam_points[:, :3]
        mean_depth = cam_points.mean(axis=0)[-1]
        mean_depths.append(mean_depth)

    return np.argsort(np.array(mean_depths))




def _get_px_coords(
        camera_pose: CameraPoseSE3,
        triangle: Triangle3d,
        cam_intrinsics: CameraIntrinsics,
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
    # cam_points = cam_points[cam_points[:, 2] > 1]

    # project to unit image plane
    projected_to_unit_plane = cam_points / cam_points[:, 2][:, np.newaxis]

    # map to pixel coordinates
    px_coords = (cam_intrinsics.get_homo_cam_coords_to_px_coords_matrix() @ projected_to_unit_plane.T).T

    return cam_points, px_coords


def _get_color(
    camera_pose: CameraPoseSE3,
    cam_points: CamCoords3d,
    light_direction: Vector3d,
) -> BGRColor:

    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()

    # we will want to compute surface normal by computing crossproduct
    # normal = triangle.get_surface_normal()
    surface_normal = np.cross(cam_points[1] - cam_points[0], cam_points[2] - cam_points[0])
    unit_surface_normal = surface_normal / np.linalg.norm(surface_normal)

    light_direction_homogenous = np.array([light_direction[0], light_direction[1], light_direction[2], 0.0])
    light_direction_in_camera = world_to_cam_flip @ SE3_inverse(camera_pose) @ light_direction_homogenous
    light_direction_in_camera = light_direction_in_camera[:3]

    light_coeff = -light_direction_in_camera[:3] @ unit_surface_normal

    # blend the color
    alpha = (light_coeff + 1) / 2

    if unit_surface_normal[2] < 0:
        from_color = np.array(BGRCuteColors.DARK_GRAY)
        to_color = np.array(BGRCuteColors.GRASS_GREEN)
    else:
        from_color = np.array(BGRCuteColors.CRIMSON)
        to_color = np.array(BGRCuteColors.DARK_GRAY)

    color = tuple(int(x) for x in (alpha * to_color + (1 - alpha) * from_color).astype(np.uint8))

    return color


def toy_render_one_triangle(
    screen: BGRImageArray,
    camera_pose: CameraPoseSE3,
    triangle: Triangle3d,
    cam_intrinsics: CameraIntrinsics,
    light_direction: Vector3d
):
    """ Just renders a triangle onto screen, ignoring whatever there might be on it already. """

    cam_points, px_coords = _get_px_coords(camera_pose, triangle, cam_intrinsics)
    color = _get_color(camera_pose, cam_points, light_direction)

    poly_coords = the_cv_flip(px_coords.round().astype(np.int32))
    cv2.fillPoly(screen, [poly_coords], color)


class TriangleRenderContext:
    mask: Array['H,W', np.bool]
    depths: Array['H,W', np.float64]
    color: BGRColor


def get_triangle_render_context(
        screen: BGRImageArray,
        camera_pose: CameraPoseSE3,
        triangle: Triangle3d,
        cam_intrinsics: CameraIntrinsics,
        light_direction: Vector3d
) -> TriangleRenderContext:
    cam_points, px_coords = _get_px_coords(camera_pose, triangle, cam_intrinsics)
    color = _get_color(camera_pose, cam_points, light_direction)
    poly_coords = the_cv_flip(px_coords.round().astype(np.int32))

    mask = np.zeros(screen.shape[:2], dtype=np.bool)
    cv2.fillPoly(mask, [poly_coords], 1)

    # get depths !
    # need better Triangle2D.to_barycentric
    pass


@attr.s(auto_attribs=True)
class MaybeTransforms:
    camera: Optional[TransformSE3] = None
    scene: Optional[TransformSE3] = None

    @classmethod
    def empty(cls):
        return cls()


def _key_to_maybe_transforms(key: int) -> MaybeTransforms:
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
        print(f"Unknown keypress {key} {chr(key)}")
        return MaybeTransforms.empty()


def get_two_triangle_scene() -> List[Triangle3d]:
    return [
        Triangle3d(np.array([
            [0.0, -1.0, -1.0, 1.0],
            [0.0,  1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0, 1.0],
        ], dtype=np.float64)),
        Triangle3d(np.array([
            [0.0,  1.0,  -1.0, 1.0],
            [0.0,  1.0,   1.0, 1.0],
            [0.0, -1.0,   1.0, 1.0],
        ], dtype=np.float64)),
    ]


def get_cube_scene() -> List[Triangle3d]:
    return [Triangle3d(homogenize(x)) for x in generate_cube_sides()]


def render_scene_naively(
    screen_h: int,
    screen_w: int,
    camera_pose: CameraPoseSE3,
    triangles: List[Triangle3d],
    cam_intrinsics: CameraIntrinsics,
    light_direction: Vector3d
):
    """ This just renders all triangles one by one.
    It resolves occlusions by trying to heuristically sort the triangles by distance to camera
    and drawing them in starting from the ones furthest from the viewer.

    """
    screen = get_canvas(shape=(screen_h, screen_w, 3), background_color=BGRCuteColors.DARK_GRAY)

    sorting_index = compute_triangle_sorting_index(camera_pose, triangles)[::-1]
    # in reality, it's more complicated - need to calculate partial occlusion
    for idx in sorting_index:
        triangle = triangles[idx]
        toy_render_one_triangle(screen, camera_pose, triangle, cam_intrinsics, light_direction)

    return screen


def render_scene_pixelwise_depth(
    screen_h: int,
    screen_w: int,
    camera_pose: CameraPoseSE3,
    triangles: List[Triangle3d],
    cam_intrinsics: CameraIntrinsics,
    light_direction: Vector3d
):
    """
Next Rendering idea:

for each triangle we have all pixels that it fills.
    For each pixel we have it's depth

for each pixel in the image, we take the nearest depth triangle and we take color from it



    """
    pass


def main():
    # I want image to be 640 by 480
    # I want it to map from 4 by 3 meters
    screen_h = 480
    screen_w = 640
    cam_intrinsics = CameraIntrinsics(fx=screen_w / 4, fy=screen_h / 3, cx=screen_w / 2, cy=screen_h / 2)
    light_direction = normalize_vector(np.array([1.0, -1.0, -8.0]))

    # looking toward +x direction in world frame, +z in camera
    camera_pose: CameraPoseSE3 = get_SE3_pose(x=-2.5)

    # triangles = get_two_triangle_scene()
    triangles = get_cube_scene()

    while True:
        screen = render_scene_naively(screen_h, screen_w, camera_pose, triangles, cam_intrinsics, light_direction)

        cv2.imshow('scene', screen)
        key = cv2.waitKey(-1)

        # mutate state based on keys
        transforms = _key_to_maybe_transforms(key)

        if transforms.scene is not None:
            triangles = [triangle.mutate(transforms.scene) for triangle in triangles]

        if transforms.camera is not None:
            camera_pose = transforms.camera @ camera_pose


if __name__ == '__main__':
    main()
