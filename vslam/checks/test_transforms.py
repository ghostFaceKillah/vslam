import numpy as np

from sim.egocentric_render import get_pixel_center_coordinates
from vslam.cam import CameraIntrinsics
from vslam.transforms import px_2d_to_world, cam_4d_to_world, world_to_cam_4d
from vslam.types import PxCoords2d


def _get_test_setup() -> tuple[CameraIntrinsics, PxCoords2d]:
    cam_intrinsics = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, screen_h=480, screen_w=640)
    #  int, x goes down, y goes right, divided by fx/fy, subtracted cx, cy
    xs_in_px = np.array([
        [89, 178],
        [0, 0],
        [479, 0],
        [0, 639],
        [479, 639],
        [240, 320],
    ], dtype=np.int64)
    return cam_intrinsics, xs_in_px


def test_px_to_world():
    cam_intrinsics, xs_in_px = _get_test_setup()
    point_in_world = px_2d_to_world(xs_in_px, cam_intrinsics)

    expected = np.array([[1, -0.44375, 0.471875,  1.], [1, -1., 0.75, 1.],
                         [1, -1., -0.746875, 1.], [1, 0.996875, 0.75, 1.],
                         [1, 0.996875, -0.746875, 1.], [1, 0., 0, 1.]], dtype=np.float64)

    assert np.allclose(point_in_world, expected)


def test_from_to_camera():
    """ Tests that cam_4d_to_world(world_to_cam_4d(x)) == x """
    cam_intrinsics, xs_in_px = _get_test_setup()
    point_in_world = px_2d_to_world(xs_in_px, cam_intrinsics)

    point_in_cam = world_to_cam_4d(point_in_world)
    point_in_world_again = cam_4d_to_world(point_in_cam)

    assert np.allclose(point_in_world, point_in_world_again)


def test_get_pixel_center_coordinates():
    cam_intrinsics = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, screen_h=480, screen_w=640)
    px_center_coords = get_pixel_center_coordinates(cam_intrinsics)

    assert np.allclose(px_center_coords[0][0], np.array([-1., -0.75], dtype=np.float32))
    assert np.allclose(px_center_coords[100][23], np.array([-0.928125, -0.4375], dtype=np.float32))
    assert np.allclose(px_center_coords[-1][-1], np.array([0.996875, 0.746875], dtype=np.float32))   # low key a bug, :)
