from typing import List

import cv2
import jax.numpy as np
import numpy as onp
from jax import jit

from sim.birds_eye_view_render import get_view_spcifier_from_scene, render_birdseye_view
from sim.clipping import ClippingSurfaces, clip_triangles
from sim.sample_scenes import get_triangles_in_sky_scene_2
from sim.sim_types import RenderTriangle3d, RenderTrianglesPointsInCam
from sim.ui import key_to_maybe_transforms
from utils.colors import BGRCuteColors
from utils.custom_types import BGRColor, Array, ImageArray
from utils.profiling import just_time
from vslam.math import normalize_vector
from vslam.poses import get_SE3_pose
from vslam.transforms import get_world_to_cam_coord_flip_matrix, SE3_inverse
from vslam.types import CameraPoseSE3, CameraIntrinsics, Vector3d, TransformSE3, ArrayOfColors


def _get_triangles_colors(
    world_to_cam_flip: TransformSE3,
    camera_pose: CameraPoseSE3,
    cam_points: RenderTrianglesPointsInCam,
    front_face_colors: ArrayOfColors,    # in future, it could be even more attributes
    back_face_colors: ArrayOfColors,
    light_direction: Vector3d,
    shade_color: BGRColor,
):
    """ Get colors of the triangles in the scene based on basic shading. """
    spanning_vectors_1 = cam_points[:, 1, :] - cam_points[:, 0, :]
    spanning_vectors_2 = cam_points[:, 2, :] - cam_points[:, 0, :]

    surface_normals = np.cross(spanning_vectors_1[:, :-1], spanning_vectors_2[:, :-1])
    inverse_norms = 1 / np.linalg.norm(surface_normals, axis=1)
    unit_surface_normals = np.einsum('i,ij->ij', inverse_norms, surface_normals)

    light_direction_homogenous = np.array([light_direction[0], light_direction[1], light_direction[2], 0.0])
    light_direction_in_camera = world_to_cam_flip @ SE3_inverse(camera_pose) @ light_direction_homogenous
    light_direction_in_camera = light_direction_in_camera[:3]

    light_coeffs = unit_surface_normals @ light_direction_in_camera
    alphas = ((light_coeffs + 1) / 2)[:, None]

    # back color blending
    # from_colors_back = np.tile(np.array(shade_color), (len(cam_points), 1))
    # to_colors_back = back_face_colors
    # colors_back = (alphas * to_colors_back + (1 - alphas) * from_colors_back).astype(np.uint8)
    #
    # from_colors_front = front_face_colors
    # to_colors_front = from_colors_back
    # colors_front = (alphas * to_colors_front + (1 - alphas) * from_colors_front).astype(np.uint8)
    #
    # front_face_filter = unit_surface_normals[:, 2] < 0
    # colors = np.where(front_face_filter[:, np.newaxis], colors_front, colors_back)

    colors = front_face_colors

    return colors


def get_pixel_center_coordinates(
    screen_h: int,
    screen_w: int,
    cam_intrinsics: CameraIntrinsics,
):
    """ Get coordinates of the center of each pixel in the camera coordinate system """
    ws = (np.arange(0, screen_w) - cam_intrinsics.cx) / cam_intrinsics.fx
    hs = (np.arange(0, screen_h) - cam_intrinsics.cy) / cam_intrinsics.fy
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
) -> ImageArray:
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
    camera_pose: CameraPoseSE3,
    sky_color: BGRColor,
    ground_color: BGRColor,
) -> ImageArray:
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
    screen_h: int,
    screen_w: int,
    camera_pose: CameraPoseSE3,
    triangles: List[RenderTriangle3d],
    cam_intrinsics: CameraIntrinsics,
    light_direction: Vector3d,
    sky_color: BGRColor,
    ground_color: BGRColor,
    shade_color: BGRColor,
    clipping_surfaces: ClippingSurfaces
):
    """
    Render scene using a variant of z-buffer algorithm.

    1) For each triangle we have all pixels that it fills.
         For each pixel we have it's depth.
         We accumulate the depth of each pixel for each triangle.

    3) for each pixel in the image, we take the nearest depth triangle and we take color from it.

    """
    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()
    cam_pose_inv = SE3_inverse(camera_pose)
    points = np.array([tri.points for tri in triangles])
    cam_points = points @ cam_pose_inv.T @ world_to_cam_flip.T

    triangle_cam_points = cam_points
    front_colors = np.array([tri.front_face_color for tri in triangles], dtype=np.uint8)
    back_colors = np.array([tri.back_face_color for tri in triangles], dtype=np.uint8)

    for clipping_surface in clipping_surfaces.to_list():
        triangle_cam_points, front_colors, back_colors = clip_triangles(
            triangle_cam_points, front_colors, back_colors, clipping_surface
        )

    px_center_coords_in_img_coords = get_pixel_center_coordinates(screen_h, screen_w, cam_intrinsics)
    bg_image = _get_background_image(px_center_coords_in_img_coords, world_to_cam_flip, camera_pose, sky_color, ground_color)

    if len(triangle_cam_points) == 0:
        return bg_image
    else:
        colors = _get_triangles_colors(
            world_to_cam_flip, camera_pose, triangle_cam_points, front_colors,
            back_colors, light_direction, shade_color
        )

        unit_depth_cam_points = triangle_cam_points[..., :-1]
        triangle_depths = unit_depth_cam_points[..., -1]
        triangles_in_img_coords = (unit_depth_cam_points / triangle_depths[..., np.newaxis])[..., :-1]

        return parallel_z_buffer_render(triangle_depths, triangles_in_img_coords, px_center_coords_in_img_coords, colors, bg_image)


def run_interactive_render_loop():
    screen_h = 480
    screen_w = 640

    sky_color = BGRCuteColors.SKY_BLUE
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
    light_direction = normalize_vector(np.array([1.0, -1.0, -8.0]))
    clipping_surfaces = ClippingSurfaces.from_screen_dimensions_and_cam_intrinsics(screen_h, screen_w, cam_intrinsics)

    # looking toward +x direction in world frame, +z in camera
    camera_pose: CameraPoseSE3 = get_SE3_pose(x=-2.5)

    # triangles = get_two_triangle_scene()
    # triangles = get_cube_scene()
    # triangles = get_triangles_in_sky_scene()
    triangles = get_triangles_in_sky_scene_2()
    view_specifier = get_view_spcifier_from_scene(triangles)

    i = 0
    while True:
        with just_time('render'):
            screen = render_scene_pixelwise_depth(
                screen_h, screen_w,
                camera_pose,
                triangles,
                cam_intrinsics,
                light_direction,
                sky_color,
                ground_color,
                shade_color,
                clipping_surfaces
            )

        bev = render_birdseye_view(
            screen_h=screen_h,
            screen_w=screen_w,
            view_specifier=view_specifier,
            camera_pose=camera_pose,
            camera_intrinsics=cam_intrinsics,
            triangles=triangles,
            bg_color=ground_color
        )

        cv2.imshow('birdseye', onp.array(bev))

        cv2.imshow('scene', onp.array(screen))
        # cv2.imwrite(f'imgs/scene_{i:04d}.png', onp.array(screen))
        key = cv2.waitKey(-1)

        # mutate state based on keys
        transforms = key_to_maybe_transforms(key)

        if transforms.scene is not None:
            triangles = [triangle.mutate(transforms.scene) for triangle in triangles]

        if transforms.camera is not None:
            camera_pose = camera_pose @ transforms.camera

        i += 1


if __name__ == '__main__':
    run_interactive_render_loop()