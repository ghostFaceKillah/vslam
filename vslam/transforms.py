from typing import Optional

import numpy as np

from vslam.cam import CameraIntrinsics
from vslam.types import PxCoords2d, ImgCoords2d, CamCoords3dHomog, TransformSE3, CameraRotationSO3, WorldCoords3D, \
    CamCoords4d


# poor man's (me) mapping from PxCoords2d to CamCoords2d
# return (px - cx) / fx, (py - cy) / fy


def px_2d_to_img_coords_2d(
    x: PxCoords2d,
    cam_intrinsics: CameraIntrinsics
) -> ImgCoords2d:
    # notice xy flip
    xs = (x[:, 1].astype(np.float64) - cam_intrinsics.cx) / cam_intrinsics.fx
    ys = (x[:, 0].astype(np.float64) - cam_intrinsics.cy) / cam_intrinsics.fy

    return np.column_stack([xs, ys])


def px_2d_to_cam_coords_3d_homo(
    xs: PxCoords2d,
    cam_intrinsics: CameraIntrinsics
) -> CamCoords3dHomog:
    # Warning! We assume no undistortion
    x_img_coords_2d = px_2d_to_img_coords_2d(xs, cam_intrinsics)
    ones = np.ones(shape=(x_img_coords_2d.shape[0], 1), dtype=np.float64)
    return np.column_stack([x_img_coords_2d, ones])


def px_2d_to_world(
    xs: PxCoords2d,
    camera_intrinsics: CameraIntrinsics,
    cam_in_world: Optional[TransformSE3] = None,
) -> WorldCoords3D:
    """ assuming the optical center is at (0, 0, 0) """
    cam_in_world = cam_in_world if cam_in_world is not None else np.eye(4, dtype=np.float64)
    xs_in_cam = px_2d_to_cam_coords_3d_homo(xs, camera_intrinsics)
    xs_in_cam_4d = homogenize(xs_in_cam)
    # world_in_flip = get_world_to_cam_coord_flip_matrix().T
    # (cam_in_world @ world_in_flip @ keypoint_in_cam_4d.T).T
    keypoint_in_world = xs_in_cam_4d @ get_world_to_cam_coord_flip_matrix() @ cam_in_world.T  # == (world_in_flip @ keypoint_in_cam_4d.T).T

    return keypoint_in_world

def cam_4d_to_world(
    xs: CamCoords4d,
    cam_in_world: Optional[TransformSE3] = None,
) -> WorldCoords3D:
    cam_in_world = cam_in_world if cam_in_world is not None else np.eye(4, dtype=np.float64)
    return xs @ get_world_to_cam_coord_flip_matrix() @ cam_in_world.T  # == (world_in_flip @ keypoint_in_cam_4d.T).T


def world_to_cam_4d(
    xs: WorldCoords3D,
    cam_in_world: Optional[TransformSE3] = None,
) -> CamCoords4d:
    cam_in_world = cam_in_world if cam_in_world is not None else np.eye(4, dtype=np.float64)
    cam_pose_inv = SE3_inverse(cam_in_world)
    return xs @ cam_pose_inv.T @ get_world_to_cam_coord_flip_matrix().T


def get_world_to_cam_coord_flip_matrix() -> TransformSE3:
    """ x = y, y = -z, z = x,
    Will take WorldCoords3D to CamCoords3d.
    aka "flip in world" matrix
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


WORLD_TO_CAM_FLIP = get_world_to_cam_coord_flip_matrix()
CAM_TO_WORLD_FLIP = SE3_inverse(WORLD_TO_CAM_FLIP)


def the_cv_flip(px_coords):
    return px_coords[:, ::-1]

def homogenize(x):
    """ e.g. (3, 1, 4) -> (3, 1, 4, 1) """
    return np.concatenate([x, np.ones(x.shape[:-1] + (1,))], axis=-1)

def dehomogenize(x):
    """ e.g. (3, 1, 4, 1)  -> (3, 1, 4) """
    return x[..., :-1]