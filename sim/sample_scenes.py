import itertools
from typing import List, Tuple

import numpy as onp
from jax import numpy as np

from sim.sim_types import RenderTriangle3d
from utils.colors import BGRCuteColors
from vslam.transforms import homogenize
from vslam.types import Vector3d


def generate_cube_sides() -> List[Tuple[Vector3d, Vector3d, Vector3d]]:
    """
    Generate 3d coordinates of triangles in 3 dimensional space that would create a 3d unit cube.
    Unit cube in the sense that it begins (and ends) at +1, -1 points, e.g. (+1, +1, +1), etc.
    """
    def _is_side_of_cube(
            a: Vector3d,
            b: Vector3d,
            c: Vector3d
    ) -> bool:
        """ Return true iff triangle lies on the side of the cube"""
        diff_ab = b - a
        diff_bc = b - c
        diff_ac = a - c

        # triangle lies on the edge if both vertex pairs are close
        m1, m2, _ = sorted([np.abs(diff).sum() / 2 for diff in [diff_ab, diff_bc, diff_ac]])
        # we would get "both the diagonals" (imagine X, we only want to get \ and not /
        # so we need to discriminate against one of the diagnoals, chosen at random
        _, _, m3 = [np.abs(diff.sum()) // 2 == 1 for diff in [diff_ab, diff_bc, diff_ac]]
        return m1 < 1.1 and m2 < 1.1 and m3 == 1

    vals = [-1, 1]
    triplets = [np.array([x, y, z]) for x in vals for y in vals for z in vals]

    # generate all possible combinations and choose only the ones that are really sides
    sides = [
        np.array(triplet, dtype=np.float64)
        for triplet in itertools.combinations(triplets, 3)
        if _is_side_of_cube(*triplet)
    ]

    return sides


def get_cube_scene() -> List[RenderTriangle3d]:
    return [RenderTriangle3d(homogenize(x)) for x in generate_cube_sides()]


def get_two_triangle_scene() -> List[RenderTriangle3d]:
    return [
        RenderTriangle3d(
            points=np.array([
                [0.0, -1.0, -1.0, 1.0],
                [0.0,  1.0, -1.0, 1.0],
                [0.0, -1.0, 1.0, 1.0],
            ], dtype=np.float64),
            front_face_color=BGRCuteColors.ORANGE,
        ),
        RenderTriangle3d(
            points=np.array([
                [0.0,  1.0,  -1.0, 1.0],
                [0.0,  1.0,   1.0, 1.0],
                [0.0, -1.0,   1.0, 1.0],
            ], dtype=np.float64),
            front_face_color=BGRCuteColors.OFF_WHITE,
        ),
    ]


def _get_triangle_from_center(
    rng: onp.random.RandomState,
    center: Vector3d,
    radius: float,
):
    pre_vertices = rng.normal(size=(3, 3))

    normalized_vertices = pre_vertices / np.linalg.norm(pre_vertices, axis=-1)[..., np.newaxis]
    vertices = (normalized_vertices * radius) + center
    return homogenize(vertices)


def _get_triangle_vertices(
    rng: onp.random.RandomState,
    radius: float,
    min_z: float,
    max_z: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> RenderTriangle3d:
    center = rng.uniform(low=(min_x, min_y, min_z), high=(max_x, max_y, max_z), size=3)
    pre_vertices = rng.normal(size=(3, 3))

    normalized_vertices = pre_vertices / np.linalg.norm(pre_vertices, axis=-1)[..., np.newaxis]
    vertices = (normalized_vertices * radius) + center
    return homogenize(vertices)


def get_triangles_in_sky_scene(
    no_small_triangles: int = 300,
    no_mid_triangles: int = 28,
    no_big_triangles: int = 8,
    rng: onp.random.RandomState = onp.random.RandomState(42),
    min_z: float = 0.5,
    max_z: float = 10.5,
    min_x: float = -20.0,
    max_x: float = 20.0,
    min_y: float = -20.0,
    max_y: float = 20.0,
) -> List[RenderTriangle3d]:
    """ Recommended soundtrack https://unknowndamage.bandcamp.com/album/drumless """

    triangles = []

    colors = list(BGRCuteColors.all().values())

    triangle_sizes = [0.5, 1.5, 15.0]

    for size, no_triangles in zip(triangle_sizes, [no_small_triangles, no_mid_triangles, no_big_triangles]):
        for _ in range(no_triangles):
            color_ix = rng.randint(len(colors))
            triangles.append(
                RenderTriangle3d(
                    _get_triangle_vertices(rng, size, min_z, max_z, min_x, max_x, min_y, max_y),
                    front_face_color=colors[color_ix],
                    back_face_color=colors[color_ix],
                )
            )

    return triangles


def get_triangles_in_sky_scene_2(
        rng: onp.random.RandomState = onp.random.RandomState(42),
        min_z: float = 0.5,
        max_z: float = 10.0,
        min_x: float = -40.0,
        max_x: float = 40.0,
        min_y: float = -40.0,
        max_y: float = 40.0,
) -> List[RenderTriangle3d]:

    xs = np.linspace(min_x, max_x, num=20)
    ys = np.linspace(min_y, max_y, num=20)
    zs = np.linspace(min_z, max_z, num=5)

    probability_exists = 0.8

    colors = [
        BGRCuteColors.OFF_WHITE, BGRCuteColors.PURPLE, BGRCuteColors.CRIMSON,
        BGRCuteColors.SALMON, BGRCuteColors.ORANGE, BGRCuteColors.SUN_YELLOW,
        BGRCuteColors.GRASS_GREEN, BGRCuteColors.TURQUOISE, BGRCuteColors.VIOLET
    ]

    sizes = [0.3, 1., 3.]

    triangles = []

    for x, y in itertools.product(xs, ys):
        if rng.random() < probability_exists:
            z = rng.choice(zs)

            size = rng.choice(sizes)

            triangle_points = _get_triangle_from_center(rng, center=np.array([x, y, z], dtype=np.float64), radius=size)
            color = colors[rng.randint(len(colors))]

            triangle = RenderTriangle3d(triangle_points, color, color)
            triangles.append(triangle)

    return triangles

if __name__ == '__main__':
    get_triangles_in_sky_scene_2()
