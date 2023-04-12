from typing import List, Tuple, Optional

import attr
import numpy as onp

from sim.sim_types import RenderTriangle3d
from utils.colors import BGRCuteColors
from utils.custom_types import PixelCoordArray, BGRColor, BGRImageArray
from utils.cv2_but_its_typed import cv2_fill_poly, cv2_line, cv2_circle
from utils.geometry import Arrow2d
from utils.image import get_canvas
from vslam.cam import CameraIntrinsics
from vslam.math import normalize_vector
from vslam.poses import SE3_pose_to_xytheta
from vslam.transforms import homogenize
from vslam.types import CameraPoseSE3, Point2d, Points2d, TransformSE3


@attr.define
class BirdseyeViewSpecifier:
    """ Parameters for rendering a birdseye view"""
    resolution: float    # how many meters per pixel
    origin: Point2d
    world_size: Tuple[float, float]     # along 2 first coordinates of world coords

    def get_pixel_size(self) -> Tuple[int, int]:
        """ Get the size of the grid in pixels """
        x, y = self.world_size
        r = self.resolution
        return int(x / r),  int(y / r)

    @classmethod
    def from_view_center(
        cls,
        view_center: Tuple[float, float],
        world_size: Tuple[float, float],
        resolution: float = 0.05,  # how many meters per pixel
    ):
        """   """

        origin = view_center[0] - world_size[1] / 2, view_center[1] - world_size[0] / 2

        return cls(
            resolution=resolution,
            origin=origin,
            world_size=world_size
        )


def bev_2d_world_to_pixel(
        world_coords: Points2d,
        view_specifier: BirdseyeViewSpecifier
) -> PixelCoordArray:
    """ Convert 2d world coordinates to pixel coordinates"""

    normalized = (world_coords - view_specifier.origin) / view_specifier.resolution
    discretized = normalized.astype(dtype=onp.int32)
    return discretized


def get_view_specifier_from_scene(
        scene: List[RenderTriangle3d],
        world_origin: Optional[Point2d] = None,
        world_size: Optional[Tuple[float, float]] = None,
        resolution: float = 0.05,
) -> BirdseyeViewSpecifier:
    """ Get a view specifier from a scene """

    if world_size is None or world_origin is None:
        all_points = onp.array([triangle.points[:, :2] for triangle in scene]).reshape(-1, 2)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)

        if world_size is None:
            world_size = (max_x - min_x, max_y - min_y)

        if world_origin is None:
            world_origin = (min_x, min_y)

    assert world_size[0] > 0
    assert world_size[1] > 0

    return BirdseyeViewSpecifier(
        resolution=resolution,
        origin=world_origin,
        world_size=world_size
    )


def draw_view_cone(
    view_specifier: BirdseyeViewSpecifier,
    camera_pose: CameraPoseSE3,
    camera_intrinsics: CameraIntrinsics,
    image: BGRImageArray,
    whiskers_length_m: float = 5.0,
    whiskers_thickness_px: int = 2,
    whiskers_color: BGRColor = BGRCuteColors.OFF_WHITE
):
    """ Draws viewport on the image """
    # TODO: draw image plane ?
    # we want to draw viewport etc
    extreme_left = - camera_intrinsics.cx / camera_intrinsics.fx
    extreme_right = (camera_intrinsics.screen_w - camera_intrinsics.cx) / camera_intrinsics.fx

    # not really cam, there is implicit coord flip here
    left_vector_in_cam = whiskers_length_m * normalize_vector(onp.array([1., extreme_right, 0.]))
    right_vector_in_cam = whiskers_length_m * normalize_vector(onp.array([1., extreme_left, 0.]))

    left_homog = camera_pose @ homogenize(left_vector_in_cam)
    right_homog = camera_pose @ homogenize(right_vector_in_cam)

    # cv2_line(image, start_point, )
    left_px = bev_2d_world_to_pixel(left_homog[:2], view_specifier)
    right_px = bev_2d_world_to_pixel(right_homog[:2], view_specifier)
    center_px = bev_2d_world_to_pixel(camera_pose[:2, -1], view_specifier)

    cv2_line(image, left_px, center_px, color=whiskers_color, thickness=whiskers_thickness_px)
    cv2_line(image, right_px, center_px, color=whiskers_color, thickness=whiskers_thickness_px)


def draw_line_on_bev(
    image: BGRImageArray,
    view_specifier: BirdseyeViewSpecifier,
    from_pt: Point2d,
    to_pt: Point2d,
    color: BGRColor,
    thickness: int = 1
):
    from_px = bev_2d_world_to_pixel(onp.array([from_pt]), view_specifier)[0]
    to_px = bev_2d_world_to_pixel(onp.array([to_pt]), view_specifier)[0]

    cv2_line(image, from_px, to_px, color=color, thickness=thickness)


def draw_circle_on_bev(
        image: BGRImageArray,
        view_specifier: BirdseyeViewSpecifier,
        pt: Point2d,
        radius: int,
        color: BGRColor,
        thickness: int = 1
):
    pt_px = bev_2d_world_to_pixel(onp.array([pt]), view_specifier)[0]
    cv2_circle(image, pt_px, radius, color, thickness)


def render_birdseye_view(
        view_specifier: BirdseyeViewSpecifier,
        camera_pose: CameraPoseSE3,
        camera_intrinsics: CameraIntrinsics,
        triangles: List[RenderTriangle3d],
        bg_color: BGRColor
) -> BGRImageArray:
    """ Renders birdseye view of the scene"""

    x_size, y_size = view_specifier.get_pixel_size()
    canvas = get_canvas(shape=(x_size, y_size, 3), background_color=bg_color)

    all_points = onp.array([triangle.points[:, :2] for triangle in triangles])
    pixel_points = bev_2d_world_to_pixel(all_points, view_specifier)

    for triangle_pts, triangle in zip(pixel_points, triangles):
        cv2_fill_poly(triangle_pts, canvas, color=triangle.front_face_color)

    draw_view_cone(view_specifier, camera_pose, camera_intrinsics, canvas)

    return canvas


@attr.define
class DisplayBirdseyeView:
    view_specifier: BirdseyeViewSpecifier
    canvas: BGRImageArray

    @classmethod
    def from_view_specifier(
        cls,
        view_specifier: BirdseyeViewSpecifier,
        ground_color: BGRColor = tuple(x - 20 for x in BGRCuteColors.CYAN)
    ):
        x_size, y_size = view_specifier.get_pixel_size()
        canvas = get_canvas(shape=(x_size, y_size, 3), background_color=ground_color)
        return cls(view_specifier, canvas)

    def draw_view_cone(
            self,
            at_pose: CameraPoseSE3,
            camera_intrinsics: CameraIntrinsics,   # viewport is for drawing viewports and not general poses
            whiskers_thickness_px: int = 2,
    ) -> None:
        """ Draw viewport """
        draw_view_cone(
            self.view_specifier,
            at_pose,
            camera_intrinsics,
            self.canvas,
            whiskers_thickness_px=whiskers_thickness_px
        )

    def draw_arrow(
            self,
            arrow: Arrow2d,
            color: BGRColor = BGRCuteColors.DARK_GRAY,
            thickness: int = 3
    ):
        for (from_pt, to_pt) in arrow.get_lines_to_draw():
            self.draw_line_2d(from_pt, to_pt, color, thickness)

    def draw_3d_pose(
        self,
        pose: TransformSE3,
        color: BGRColor = BGRCuteColors.DARK_GRAY,
        arrow_length: float = 0.15,
        thickness: int = 2
    ):
        pose_2d = SE3_pose_to_xytheta(pose)
        arrow = Arrow2d.from_length_and_origin(origin=pose_2d[:2], length=arrow_length, orientation=pose_2d[2])
        self.draw_arrow(arrow, color, thickness)

    def draw_line_2d(
            self,
            from_pt: Point2d,
            to_pt: Point2d,
            color: BGRColor,
            thickness: int = 1
    ):
        draw_line_on_bev(
            self.canvas,
            self.view_specifier,
            from_pt,
            to_pt,
            color,
            thickness
        )

    def draw_circle(
            self,
            pt: Point2d,
            color: BGRColor,
            radius: int = 1,
            thickness: int = 1
    ):
        draw_circle_on_bev(
            self.canvas,
            self.view_specifier,
            pt,
            radius,
            color,
            thickness
        )

    def draw_triangles(
        self,
        triangles: List[RenderTriangle3d],
    ) -> None:
        """ aka draw scene """

        all_points = onp.array([triangle.points[:, :2] for triangle in triangles])
        pixel_points = bev_2d_world_to_pixel(all_points, self.view_specifier)

        for triangle_pts, triangle in zip(pixel_points, triangles):
            cv2_fill_poly(triangle_pts, self.canvas, color=triangle.front_face_color)

    def get_image(self) -> BGRImageArray:
        return self.canvas