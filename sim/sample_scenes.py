import itertools
from typing import List, Tuple

from jax import numpy as np

from sim.sim_types import RenderTriangle3d
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
        RenderTriangle3d(np.array([
            [0.0, -1.0, -1.0, 1.0],
            [0.0,  1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0, 1.0],
        ], dtype=np.float64)),
        RenderTriangle3d(np.array([
            [0.0,  1.0,  -1.0, 1.0],
            [0.0,  1.0,   1.0, 1.0],
            [0.0, -1.0,   1.0, 1.0],
        ], dtype=np.float64)),
    ]