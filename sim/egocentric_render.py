from typing import List

import jax.numpy as np
from jax import jit

from sim.clipping import ClippingSurfaces, clip_triangles
from sim.sim_types import RenderTriangle3d
from utils.custom_types import BGRColor, Array, JaxImageArray
from vslam.cam import CameraIntrinsics
from vslam.transforms import get_world_to_cam_coord_flip_matrix, world_to_cam_4d
from vslam.types import TransformSE3, Vector3d, ArrayOfColors


def get_pixel_center_coordinates(cam_intrinsics: CameraIntrinsics) -> Array['H,W,2', np.float32]:
    """ Get coordinates of the center of each pixel in the image coordinate system """
    ws = (np.arange(0, cam_intrinsics.screen_w) - cam_intrinsics.cx) / cam_intrinsics.fx
    hs = (np.arange(0, cam_intrinsics.screen_h) - cam_intrinsics.cy) / cam_intrinsics.fy
    px_center_coords_in_cam = np.stack(np.meshgrid(ws, hs), axis=-1)
    return px_center_coords_in_cam


def compute_barycentric_coordinates_of_pixels(
    triangles_in_image,
    px_center_coords_in_cam
):
    """
     For each pixel center, express its coordinates in barycentric coordinates of the all visible triangles.
     See https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Edge_approach for details of the algorithm.
      """
    T = np.zeros(shape=(len(triangles_in_image), 2, 2), dtype=np.float32)

    x_1 = triangles_in_image[:, 0, 0]
    y_1 = triangles_in_image[:, 0, 1]
    x_2 = triangles_in_image[:, 1, 0]
    y_2 = triangles_in_image[:, 1, 1]
    x_3 = triangles_in_image[:, 2, 0]
    y_3 = triangles_in_image[:, 2, 1]
    r_3 = triangles_in_image[:, 2, :]

    T = T.at[:, 0, 0].set(x_1 - x_3)
    T = T.at[:, 0, 1].set(x_2 - x_3)
    T = T.at[:, 1, 0].set(y_1 - y_3)
    T = T.at[:, 1, 1].set(y_2 - y_3)

    T_inv = np.linalg.inv(T)

    normalized_px_coords = px_center_coords_in_cam[:, :, np.newaxis, :] - r_3[np.newaxis, np.newaxis, :, :]
    bary_partial = np.einsum('hwnc,nbc->hwnb', normalized_px_coords, T_inv)
    third_coordinate = 1 - bary_partial.sum(axis=-1)
    bary = np.block([bary_partial, third_coordinate[..., np.newaxis]])

    return bary


@jit
def parallel_z_buffer_render(
    triangle_depths: Array['N,3', np.float32],   # N x 3  (for each triangle, depths of the 3 vertices in camera coordinates)
    triangles_in_img_coords: Array['N,3,2', np.float32],  # N x 3 x 2  (for each triangle, 2D coordinates of the 3 vertices in image coordinates)
    px_center_coords_in_img_coords: Array['H,W,2', np.float32],  # H x W x 2  (for each pixel, 2D coordinates of the pixel center in camera coordinates)
    lighting_aware_colors: ArrayOfColors, # N x 3 (for each triangle, color of the face with lighting already taken into account)
    bg_img: Array['H,W,3', np.uint8],  # H x W x 3  (background image)
) -> JaxImageArray:
    """
    Render triangles using a variant of z-buffer algorithm.

    For each pixel, we express position of the center of this pixel in barycentric coordinates of every triangle.
    That is, each pixel center is a linear combination of triangle vertices.

    Thanks to that we know if given pixel is inside the triangle and, if yes, at what depth the
    center-of-pixel ray intersects the triangle. In this way for each pixel we can compute the closest
    ray-intersecting triangle and take it's lighting-aware color as the pixels color.
    """

    bary = compute_barycentric_coordinates_of_pixels(triangles_in_img_coords, px_center_coords_in_img_coords)

    # N x H x W x 1, so pretty big
    # we are butchering the z-buffer implementation a bit - so much memory shouldn't be neccessary, but (!)
    # the trade-off that we are taking here is that we avoid for loops, which are slow in python
    depth_per_pixel_per_triangle = (bary * triangle_depths[np.newaxis, np.newaxis, ...]).sum(axis=-1)

    inside_triangle_pixel_filter = np.all(bary > 0, axis=-1)

    # TODO: remove - move to clipping
    in_front_of_image_plane = depth_per_pixel_per_triangle > 1.0

    est_depth_per_pixel_per_triangle = np.where(
        inside_triangle_pixel_filter & in_front_of_image_plane,
        depth_per_pixel_per_triangle,
        np.inf
    )

    # here's the z-buffer for loop that we avoid
    best_triangle_idx = np.argmin(est_depth_per_pixel_per_triangle, axis=-1)
    px_with_any_triangle = np.any(inside_triangle_pixel_filter & in_front_of_image_plane, axis=-1)

    image = np.where(
        px_with_any_triangle[..., np.newaxis],
        lighting_aware_colors[best_triangle_idx],
        bg_img
    )

    return image


def _get_background_image(
    px_center_coords_in_img_coords: Array['H,W,2', np.float32],
    world_to_cam_flip: Array['4,4', np.float32],
    camera_pose: TransformSE3,
    sky_color: BGRColor,
    ground_color: BGRColor,
) -> JaxImageArray:
    """ Get background image for the scene. """

    px_center_coords_in_world = np.concatenate([px_center_coords_in_img_coords, np.ones_like(px_center_coords_in_img_coords)], axis=-1) @ world_to_cam_flip @ camera_pose.T
    px_z_in_world = px_center_coords_in_world[..., 2]
    optical_center_z_in_world = camera_pose[2, 3]
    px_looks_toward = px_z_in_world - optical_center_z_in_world

    sky_color_arr = np.array(sky_color, dtype=np.uint8)
    ground_color_arr = np.array(ground_color, dtype=np.uint8)

    bg_image = np.where(
        px_looks_toward[..., np.newaxis] > 0,
        sky_color_arr[np.newaxis, np.newaxis, :],
        ground_color_arr[np.newaxis, np.newaxis, :]
    )

    return bg_image


def render_scene_pixelwise_depth(
    camera_pose: TransformSE3,
    triangles: List[RenderTriangle3d],
    cam_intrinsics: CameraIntrinsics,
    sky_color: BGRColor,
    ground_color: BGRColor,
    clipping_surfaces: ClippingSurfaces
) -> JaxImageArray:
    """
    Render scene using a variant of z-buffer algorithm.

    1) For each triangle we have all pixels that it fills.
         For each pixel we have it's depth.
         We accumulate the depth of each pixel for each triangle.

    3) for each pixel in the image, we take the nearest depth triangle and we take color from it.

    """
    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()

    points = np.array([tri.points for tri in triangles])
    points_in_cam = world_to_cam_4d(points, camera_pose)

    triangle_cam_points = points_in_cam
    front_colors = np.array([tri.front_face_color for tri in triangles], dtype=np.uint8)
    back_colors = np.array([tri.back_face_color for tri in triangles], dtype=np.uint8)

    for clipping_surface in clipping_surfaces.to_list():
        triangle_cam_points, front_colors, back_colors = clip_triangles(
            triangle_cam_points, front_colors, back_colors, clipping_surface
        )

    px_center_coords_in_img_coords = get_pixel_center_coordinates(cam_intrinsics)
    bg_image = _get_background_image(px_center_coords_in_img_coords, world_to_cam_flip, camera_pose, sky_color, ground_color)

    if len(triangle_cam_points) == 0:
        return bg_image
    else:
        colors = front_colors
        unit_depth_cam_points = triangle_cam_points[..., :-1]
        triangle_depths = unit_depth_cam_points[..., -1]
        triangles_in_img_coords = (unit_depth_cam_points / triangle_depths[..., np.newaxis])[..., :-1]

        return parallel_z_buffer_render(triangle_depths, triangles_in_img_coords, px_center_coords_in_img_coords, colors, bg_image)
