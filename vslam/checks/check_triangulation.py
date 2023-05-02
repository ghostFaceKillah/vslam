import numpy as np

from vslam.transforms import get_world_to_cam_coord_flip_matrix
from vslam.triangulation import naive_triangulation


def check_triangulation_known_case():
    point_in_world = np.array([8., -0.8, 1., 1.])
    world_in_base = get_world_to_cam_coord_flip_matrix()
    point_in_base_4d = world_in_base @ point_in_world
    # point_in_base_3d = point_in_base_4d[:3]
    point_in_left_3d = np.array([.2, -1., 8.])
    point_in_right_3d = np.array([-1.8, -1., 8.])
    t = np.array([-2., 0., 0.])

    # assert s_2 * x_2 = s_1 * R * x_1 + t
    assert np.allclose(point_in_right_3d, point_in_left_3d + t)
    left_in_right = np.array([
        [1, 0, 0, -2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    depth = naive_triangulation(
        points_in_cam_one=np.array([point_in_left_3d]) / 8.,
        points_in_cam_two=np.array([point_in_right_3d]) / 8.,
        cam_one_in_two=left_in_right
    )

    assert depth[0].depth_or_none == 8.


if __name__ == '__main__':
    check_triangulation_known_case()
