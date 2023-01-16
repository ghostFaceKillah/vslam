"""
Small demo of naive rendering


"""
from typing import List, Tuple

import cv2
import jax.numpy as np
import numpy as onp
from jax import jit

from sim.clipping import ClippingSurfaces
from sim.sample_scenes import get_cube_scene
from sim.sim_types import RenderTriangle3d, RenderTrianglesPointsInCam
from sim.ui import key_to_maybe_transforms
from utils.colors import BGRCuteColors
from utils.custom_types import BGRColor, Array, ImageArray
from utils.profiling import just_time
from vslam.math import normalize_vector
from vslam.poses import get_SE3_pose
from vslam.transforms import get_world_to_cam_coord_flip_matrix, SE3_inverse
from vslam.types import CameraPoseSE3, CameraIntrinsics, Vector3d, TransformSE3, CamCoords3d, ArrayOfColors


def _get_triangles_colors(
    world_to_cam_flip: TransformSE3,
    camera_pose: CameraPoseSE3,
    cam_points: RenderTrianglesPointsInCam,
    front_face_colors: ArrayOfColors,    # in future, it could be even more attributes
    back_face_colors: ArrayOfColors,
    light_direction: Vector3d,
    shade_color: BGRColor
):
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
    from_colors_back = np.tile(np.array(shade_color), (len(cam_points), 1))
    to_colors_back = back_face_colors
    colors_back = (alphas * to_colors_back + (1 - alphas) * from_colors_back).astype(np.uint8)

    from_colors_front = front_face_colors
    to_colors_front = from_colors_back
    colors_front = (alphas * to_colors_front + (1 - alphas) * from_colors_front).astype(np.uint8)

    front_face_filter = unit_surface_normals[:, 2] < 0
    colors = np.where(front_face_filter[:, np.newaxis], colors_front, colors_back)

    return colors


def get_pixel_center_coordinates(
    screen_h: int,
    screen_w: int,
    cam_intrinsics: CameraIntrinsics,
):
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
    lighting_aware_colors: ArrayOfColors  # N x 3 (for each triangle, color of the face with lighting already taken into account)
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

    bg_color = np.array(BGRCuteColors.DARK_GRAY, dtype=np.uint8)
    # we could pass horizon image

    image = np.where(
        px_with_any_triangle[..., np.newaxis],
        lighting_aware_colors[best_triangle_idx],
        bg_color[np.newaxis, np.newaxis, :]
    )

    return image


def _compute_intersection(
    starting_points: CamCoords3d,
    direction_vectors: CamCoords3d,
    surface_normal: Vector3d
) -> CamCoords3d:
    # t = - <N, A> / <N, B-A>
    # what if <N, B-A> is zero ??
    # TODO!
    t = - (starting_points[..., :-1] @ surface_normal) / (direction_vectors[..., :-1] @ surface_normal)
    return starting_points + t[:, np.newaxis] * direction_vectors


def _clip_triangles_with_one_vertex_visible(
    cam_points: CamCoords3d,
        front_face_colors: ArrayOfColors,    # in future, it could be even more attributes
        back_face_colors: ArrayOfColors,
    clipping_surface_normal: Vector3d,
    no_vertices_visible,
    signed_dists
):
    # need to put the visible vertex in known place in the array
    clip_triangles_filter = no_vertices_visible == 1

    wanted_move = (3 - np.argmax(signed_dists[clip_triangles_filter], axis=-1)) % 3
    ix_array = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    the_glorious_roll = ix_array[wanted_move]

    cam_points_in_need_of_clipping = np.array(
        [triangle_vertices[right_order]
         for triangle_vertices, right_order in zip(cam_points[clip_triangles_filter], the_glorious_roll)]
    )

    a = cam_points_in_need_of_clipping[:, 0, :]
    b_minus_a = cam_points_in_need_of_clipping[:, 1, :] - cam_points_in_need_of_clipping[:, 0, :]
    c_minus_a = cam_points_in_need_of_clipping[:, 2, :] - cam_points_in_need_of_clipping[:, 0, :]

    b_prime = _compute_intersection(a, b_minus_a, clipping_surface_normal)
    c_prime = _compute_intersection(a, c_minus_a, clipping_surface_normal)

    clipped_triangles = np.stack([a, b_prime, c_prime], axis=1)
    new_front_face_colors = front_face_colors[clip_triangles_filter]
    new_back_face_colors = back_face_colors[clip_triangles_filter]

    return clipped_triangles, new_front_face_colors, new_back_face_colors


def clip_two_vertices_visible_triangles(
        cam_points: CamCoords3d,
        front_face_colors: ArrayOfColors,    # in future, it could be even more attributes
        back_face_colors: ArrayOfColors,
        clipping_surface_normal: Vector3d,
        no_vertices_visible,
        signed_dists
):
    """ https://gabrielgambetta.com/computer-graphics-from-scratch/11-clipping.html """
    # need to put the visible vertex in known place in the array
    clip_triangles_filter = no_vertices_visible == 2

    wanted_move = (2 - np.argmin(signed_dists[clip_triangles_filter], axis=-1))
    ix_array = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])

    the_glorious_roll = ix_array[wanted_move]

    cam_points_in_need_of_clipping = np.array(
        [triangle_vertices[right_order]
         for triangle_vertices, right_order in zip(cam_points[clip_triangles_filter], the_glorious_roll)]
    )

    a = cam_points_in_need_of_clipping[:, 0, :]
    b = cam_points_in_need_of_clipping[:, 1, :]
    c = cam_points_in_need_of_clipping[:, 2, :]

    a_prime = _compute_intersection(a, c - a, clipping_surface_normal)
    b_prime = _compute_intersection(b, c - b, clipping_surface_normal)

    clipped_triangles_one = np.stack([a, b, a_prime], axis=1)
    clipped_triangles_two = np.stack([a_prime, b, b_prime], axis=1)

    # [Triangle(A, B, A'), Triangle(A', B, B')]
    new_front_face_colors = front_face_colors[clip_triangles_filter]
    new_back_face_colors = back_face_colors[clip_triangles_filter]

    triangles = np.concatenate([clipped_triangles_one, clipped_triangles_two])
    new_front_face_colors = np.concatenate([new_front_face_colors, new_front_face_colors])
    new_back_face_colors = np.concatenate([new_back_face_colors, new_back_face_colors])

    return triangles, new_front_face_colors, new_back_face_colors


def clip_triangles(
    cam_points: CamCoords3d,
    front_face_colors: ArrayOfColors,    # in future, it could be even more attributes
    back_face_colors: ArrayOfColors,
    clipping_surface_normal: Vector3d,
) -> Tuple[CamCoords3d, ArrayOfColors, ArrayOfColors]:
    """
    Clip triangles against a plane defined by a normal vector.
    https://gabrielgambetta.com/computer-graphics-from-scratch/11-clipping.html
    """
    signed_dists = np.einsum('ntd,d->nt', cam_points[..., :-1], clipping_surface_normal)
    no_vertices_visible = (signed_dists > 0).sum(axis=-1)

    resu_triangles = []
    resu_front_face_colors = []
    resu_back_face_colors = []

    if (no_vertices_visible == 3).any():
        # fully visible triangles, no clipping needed
        triangles_filter = no_vertices_visible == 3
        resu_triangles.append(cam_points[triangles_filter])
        resu_front_face_colors.append(front_face_colors[triangles_filter])
        resu_back_face_colors.append(back_face_colors[triangles_filter])

    if (no_vertices_visible == 1).any():
        clipped_triangles, clipped_front_face_colors, clipped_back_face_colors = _clip_triangles_with_one_vertex_visible(
            cam_points, front_face_colors, back_face_colors,
            clipping_surface_normal, no_vertices_visible, signed_dists
        )
        resu_triangles.append(clipped_triangles)
        resu_front_face_colors.append(clipped_front_face_colors)
        resu_back_face_colors.append(clipped_back_face_colors)

    if (no_vertices_visible == 2).any():
        clipped_triangles, clipped_front_face_colors, clipped_back_face_colors = clip_two_vertices_visible_triangles(
            cam_points, front_face_colors, back_face_colors,
            clipping_surface_normal, no_vertices_visible, signed_dists
        )
        resu_triangles.append(clipped_triangles)
        resu_front_face_colors.append(clipped_front_face_colors)
        resu_back_face_colors.append(clipped_back_face_colors)

    if len(resu_triangles) > 0:
        triangles = np.concatenate(resu_triangles)
        front_face_colors = np.concatenate(resu_front_face_colors)
        back_face_colors = np.concatenate(resu_back_face_colors)
        assert len(triangles) == len(front_face_colors) == len(back_face_colors)

        return triangles, front_face_colors, back_face_colors
    else:
        return np.empty((0, 3, 4)), np.empty((0, 3, 3)), np.empty((0, 3, 3))


def render_scene_pixelwise_depth(
    screen_h: int,
    screen_w: int,
    camera_pose: CameraPoseSE3,
    triangles: List[RenderTriangle3d],
    cam_intrinsics: CameraIntrinsics,
    light_direction: Vector3d,
    shade_color: BGRColor,
    clipping_surfaces: ClippingSurfaces
):
    """
    Next Rendering idea:

    1) Maybe decide to not render some of the triangles based on visibility

    2) for each triangle we have all pixels that it fills.
         For each pixel we have it's depth

    3) for each pixel in the image, we take the nearest depth triangle and we take color from it

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

    if len(triangle_cam_points) == 0:
        return np.ones((screen_h, screen_w, 3), dtype=np.uint8) * np.array(shade_color, dtype=np.uint8)
    else:
        colors = _get_triangles_colors(world_to_cam_flip, camera_pose, triangle_cam_points, front_colors,
                                       back_colors, light_direction, shade_color)

        px_center_coords_in_img_coords = get_pixel_center_coordinates(screen_h, screen_w, cam_intrinsics)

        unit_depth_cam_points = triangle_cam_points[..., :-1]
        triangle_depths = unit_depth_cam_points[..., -1]
        triangles_in_img_coords = (unit_depth_cam_points / triangle_depths[..., np.newaxis])[..., :-1]

        return parallel_z_buffer_render(triangle_depths, triangles_in_img_coords, px_center_coords_in_img_coords, colors)


def main():
    screen_h = 480
    screen_w = 640

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
    triangles = get_cube_scene()

    while True:
        with just_time('render'):
            screen = render_scene_pixelwise_depth(
                screen_h, screen_w,
                camera_pose,
                triangles,
                cam_intrinsics,
                light_direction,
                shade_color,
                clipping_surfaces
            )

        cv2.imshow('scene', onp.array(screen))
        key = cv2.waitKey(-1)

        # mutate state based on keys
        transforms = key_to_maybe_transforms(key)

        if transforms.scene is not None:
            triangles = [triangle.mutate(transforms.scene) for triangle in triangles]

        if transforms.camera is not None:
            camera_pose = camera_pose @ transforms.camera


if __name__ == '__main__':
    main()