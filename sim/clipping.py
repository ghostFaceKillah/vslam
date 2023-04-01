from typing import List, Tuple

import attr
import numpy as onp
from jax import numpy as np

from utils.custom_types import Array
from vslam.cam import CameraIntrinsics
from vslam.math import normalize_vector
from vslam.types import Vector3d, CamCoords3d, ArrayOfColors


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

        a = normalize_vector(onp.array([extreme_right, 0, 1.]))
        b = onp.array([0.0, -1.0, 0.0])
        right_clipping_surface_normal = onp.cross(b, a)

        a = normalize_vector(onp.array([extreme_left, 0, 1.]))
        b = onp.array([0.0, 1.0, 0.0])
        left_clipping_surface_normal = onp.cross(b, a)

        a = normalize_vector(onp.array([0., extreme_up, 1.]))
        b = onp.array([-1.0, 0.0, 0.0])
        upper_clipping_surface_normal = onp.cross(b, a)

        a = normalize_vector(onp.array([0., extreme_down, 1.]))
        b = onp.array([1.0, 0.0, 0.0])
        lower_clipping_surface_normal = onp.cross(b, a)

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


def _compute_intersection(
    starting_points: CamCoords3d,
    direction_vectors: CamCoords3d,
    surface_normal: Vector3d
) -> CamCoords3d:
    """ Compute point of intersection between a plane and collection of vectors with given starting points
    and given directions. Used in clipping (during rendering) for computing clipping points
    for many triangles at the same time.
    See https://gabrielgambetta.com/computer-graphics-from-scratch/11-clipping.html
    """
    # TODO:  t = - <N, A> / <N, B-A>, but  what if <N, B-A> is zero ??
    t = - (starting_points[..., :-1] @ surface_normal) / (direction_vectors[..., :-1] @ surface_normal)
    return starting_points + t[:, np.newaxis] * direction_vectors


def _clip_triangles_with_one_vertex_visible(
    cam_points: CamCoords3d,
    front_face_colors: ArrayOfColors,    # in future, it could be even more attributes
    back_face_colors: ArrayOfColors,
    clipping_surface_normal: Vector3d,
    no_vertices_visible: Array['N', np.int32],
    signed_dists: Array['N,3', np.float32],
) -> Tuple[CamCoords3d, ArrayOfColors, ArrayOfColors]:
    """ Clip triangles for which we know exactly one vertex is visible (wrt given clipping surface)
        See https://gabrielgambetta.com/computer-graphics-from-scratch/11-clipping.html """
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


def _clip_two_vertices_visible_triangles(
        cam_points: CamCoords3d,
        front_face_colors: ArrayOfColors,    # in future, it could be even more attributes
        back_face_colors: ArrayOfColors,
        clipping_surface_normal: Vector3d,
        no_vertices_visible: Array['N', np.int32],
        signed_dists: Array['N,3', np.float32],
) -> Tuple[CamCoords3d, ArrayOfColors, ArrayOfColors]:
    """ Clip triangles for which we know exactly two vertices are visible (wrt given clipping surface)
        See https://gabrielgambetta.com/computer-graphics-from-scratch/11-clipping.html """
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

    # [Triangle(A, B, A'), Triangle(A', B, B')]
    clipped_triangles_one = np.stack([a, b, a_prime], axis=1)
    clipped_triangles_two = np.stack([a_prime, b, b_prime], axis=1)

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
        clipped_triangles, clipped_front_face_colors, clipped_back_face_colors = _clip_two_vertices_visible_triangles(
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