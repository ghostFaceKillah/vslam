import numpy as np

from vslam.triangulation import naive_triangulation


def test_triangulation_known_case():
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
