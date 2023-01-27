from typing import List, Tuple, Optional

import attr
import cv2
import numpy as onp

from sim.sample_scenes import get_triangles_in_sky_scene
from sim.sim_types import RenderTriangle3d
from sim.ui import key_to_maybe_transforms
from utils.colors import BGRCuteColors
from utils.custom_types import PixelCoordArray, BGRColor, BGRImageArray
from utils.cv2_but_its_typed import cv2_fill_poly, cv2_line
from utils.image import get_canvas
from utils.profiling import just_time
from vslam.math import normalize_vector
from vslam.poses import get_SE3_pose
from vslam.transforms import homogenize
from vslam.types import CameraPoseSE3, CameraIntrinsics, Point2d, Points2d


@attr.define
class BirdseyeViewParams:
    resolution: float    # how many meters per pixel
    origin: Point2d
    world_size: Tuple[float, float]     # along 2 first coordinates of world coords

    def get_pixel_size(self) -> Tuple[int, int]:
        x, y = self.world_size
        r = self.resolution
        return int(x / r),  int(y / r)


def bev_2d_world_to_pixel(
        world_coords: Points2d,
        view_specifier: BirdseyeViewParams
) -> PixelCoordArray:

    normalized = (world_coords - view_specifier.origin) / view_specifier.resolution
    discretized = normalized.astype(dtype=onp.int32)
    return discretized



def bev_pixel_to_2d_world(pixel_coords: PixelCoordArray) -> Points2d:
    pass


def get_view_spcifier_from_scene(
        scene: List[RenderTriangle3d],
        world_origin: Optional[Point2d] = None,
        world_size: Optional[Tuple[float, float]] = None,
        resolution: float = 0.05,
):

    if world_size is None or world_origin is None:
        all_points = onp.array([triangle.points[:, :2] for triangle in scene]).reshape(-1, 2)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)

        if world_size is None:
            world_size = (max_x - min_x, max_y - min_y)

        if world_origin is None:
            world_origin = (min_x, min_y)

    return BirdseyeViewParams(
        resolution=resolution,
        origin=world_origin,
        world_size=world_size
    )


def draw_viewport(
    screen_h: int,
    screen_w: int,
    view_specifier: BirdseyeViewParams,
    camera_pose: CameraPoseSE3,
    camera_intrinsics: CameraIntrinsics,
    image: BGRImageArray,
    whiskers_length_m: float = 3.0
):
    # we want to draw viewport etc
    extreme_left = - camera_intrinsics.cx / camera_intrinsics.fx
    extreme_right = (screen_w - camera_intrinsics.cx) / camera_intrinsics.fx

    # not really cam, there is implicit coord flip here
    left_vector_in_cam = whiskers_length_m * normalize_vector(onp.array([1., extreme_right, 0.]))
    right_vector_in_cam = whiskers_length_m * normalize_vector(onp.array([1., extreme_left, 0.]))

    left_homog = camera_pose @ homogenize(left_vector_in_cam)
    right_homog = camera_pose @ homogenize(right_vector_in_cam)

    # cv2_line(image, start_point, )
    left_px = bev_2d_world_to_pixel(left_homog[:2], view_specifier)
    right_px = bev_2d_world_to_pixel(right_homog[:2], view_specifier)
    center_px = bev_2d_world_to_pixel(camera_pose[:2, -1], view_specifier)

    cv2_line(image, left_px, center_px, color=BGRCuteColors.OFF_WHITE, thickness=1)
    cv2_line(image, right_px, center_px, color=BGRCuteColors.OFF_WHITE, thickness=1)


def render_birdseye_view(
        screen_h: int,
        screen_w: int,
        view_specifier: BirdseyeViewParams,
        camera_pose: CameraPoseSE3,
        camera_intrinsics: CameraIntrinsics,
        triangles: List[RenderTriangle3d],
        bg_color: BGRColor
) -> BGRImageArray:

    # make appropriate costmap bgr array
    x_size, y_size = view_specifier.get_pixel_size()
    canvas = get_canvas(shape=(x_size, y_size, 3), background_color=bg_color)

    all_points = onp.array([triangle.points[:, :2] for triangle in triangles])
    pixel_points = bev_2d_world_to_pixel(all_points, view_specifier)

    # (336, 3, 2)
    for triangle_pts, triangle in zip(pixel_points, triangles):
        cv2_fill_poly(triangle_pts, canvas, color=triangle.front_face_color)

    draw_viewport(screen_h, screen_w, view_specifier, camera_pose, camera_intrinsics, canvas)

    return canvas


if __name__ == '__main__':

    screen_h = 480
    screen_w = 640

    ground_color = tuple(x - 20 for x in BGRCuteColors.CYAN)

    # higher f_mod -> less distortion, less field of view
    f_mod = 2.0

    shade_color = BGRCuteColors.DARK_GRAY

    cam_intrinsics = CameraIntrinsics(
        fx=screen_w / 4 * f_mod,
        fy=screen_h / 3 * f_mod,
        cx=screen_w / 2,
        cy=screen_h / 2,
    )
    # clipping_surfaces = ClippingSurfaces.from_screen_dimensions_and_cam_intrinsics(screen_h, screen_w, cam_intrinsics)

    # looking toward +x direction in world frame, +z in camera
    camera_pose: CameraPoseSE3 = get_SE3_pose(x=-2.5)

    # triangles = get_two_triangle_scene()
    # triangles = get_cube_scene()
    triangles = get_triangles_in_sky_scene()

    view_specifier = get_view_spcifier_from_scene(triangles)


    while True:
        with just_time('render'):
            screen = render_birdseye_view(
                view_specifier=view_specifier,
                camera_pose=camera_pose,
                camera_intrinsics=cam_intrinsics,
                triangles=triangles,
                bg_color=ground_color
            )

        cv2.imshow('scene', onp.array(screen))
        # cv2.imwrite(f'imgs/scene_{i:04d}.png', onp.array(screen))
        key = cv2.waitKey(-1)

        # mutate state based on keys
        transforms = key_to_maybe_transforms(key)

        if transforms.scene is not None:
            triangles = [triangle.mutate(transforms.scene) for triangle in triangles]

        if transforms.camera is not None:
            camera_pose = camera_pose @ transforms.camera

