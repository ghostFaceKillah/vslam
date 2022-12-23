import numpy as np

from vslam.types import CameraIntrinsics, PxCoords2d, ImgCoords2d, CamCoords3dHomog, TransformSE3, CameraRotationSO3


# poor man's (me) mapping from PxCoords2d to CamCoords2d
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


def get_world_to_cam_coord_flip_matrix() -> TransformSE3:
    """ x = y, y = -z, z = x,
    Will take WorldCoords3D to CamCoords3d.
    """
    return np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def SO3_inverse(R: CameraRotationSO3) -> CameraRotationSO3:
    """Inverts the rotation."""
    return R.T


def SE3_inverse(T: TransformSE3) -> TransformSE3:
    """Returns the inverse of the transformation."""
    tf = np.eye(4, dtype=np.float64)
    R_inv = T[:3, :3].T
    t = T[:3, 3]
    tf[0:3, 0:3] = R_inv
    tf[0:3, 3] = - R_inv @ t
    return tf


def the_cv_flip(px_coords):
    return px_coords[:, ::-1]