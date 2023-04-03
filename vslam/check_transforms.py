import numpy as np

from sim.egocentric_render import get_pixel_center_coordinates
from vslam.cam import CameraIntrinsics
from vslam.transforms import px_2d_to_world, cam_4d_to_world, world_to_cam_4d


def check_from_to_world():
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
    point_in_world = px_2d_to_world(xs_in_px, cam_intrinsics)

    px_center_coords = get_pixel_center_coordinates(cam_intrinsics)

    # t = 89, 178; px_center_coords[t], px_2d_to_world(np.array([t]), cam_intrinsics)
    pass


def check_from_to_camera():
    cam_intrinsics = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, screen_h=480, screen_w=640)

    xs_in_px = np.array([
        [89, 178],
        [0, 0],
        [479, 0],
        [0, 639],
        [479, 639],
        [240, 320],
    ], dtype=np.int64)
    point_in_world = px_2d_to_world(xs_in_px, cam_intrinsics)

    point_in_cam = world_to_cam_4d(point_in_world)
    point_in_world_again = cam_4d_to_world(point_in_cam)

    assert np.allclose(point_in_world, point_in_world_again)


def check_rendering():
    # above, we have points that are mapped px -> world -> cam -> world
    """
    TODO
    render known point.
    see if it's good.

    :return:
    """

    pass


def check_rendering_pipeline_px_center_coords():
    """ Make sure that px center coordinates add up to thing that's reasonable hehe """
    cam_intrinsics = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, screen_h=480, screen_w=640)
    px_center_coords = get_pixel_center_coordinates(cam_intrinsics)
    pass

if __name__ == '__main__':
    check_from_to_camera()
