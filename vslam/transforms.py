import numpy as np

from vslam.types import CameraIntrinsics, PxCoords2d, ImgCoords2d, CamCoords3dHomog


# poor man's mapping from PxCoords2d to CamCoords2d
# return (px - cx) / fx, (py - cy) / fy


def px_2d_to_cam_coords_2d(
    x: PxCoords2d,
    cam_intrinsics: CameraIntrinsics
) -> ImgCoords2d:
    # notice xy flip
    xs = (x[:, 1].astype(np.float64) - cam_intrinsics.cx) / cam_intrinsics.fx
    ys = (x[:, 0].astype(np.float64) - cam_intrinsics.cy) / cam_intrinsics.fy

    return np.column_stack([xs, ys])


def px_2d_to_cam_coords_3d_homo(
    x: PxCoords2d,
    cam_intrinsics: CameraIntrinsics
) -> CamCoords3dHomog:
    # Warning! We assume no undistortion
    x_img_coords_2d = px_2d_to_cam_coords_2d(x, cam_intrinsics)
    ones = np.ones(shape=(x_img_coords_2d.shape[0], 1), dtype=np.float64)
    return np.column_stack([x_img_coords_2d, ones])
