from typing import Optional

import numpy as np

from liegroups.numpy.se3 import SE3Matrix
from utils.custom_types import Array
from vslam.types import CamFlippedWorldCoords3D, ImgCoords2d, CameraPoseSE3, ReprojectionErrorVector

ErrorSe3PoseJacobian = Array['6,2', np.float64]


def estimate_J_numerically(
        point_3d: CamFlippedWorldCoords3D,
        point_2d: ImgCoords2d,
        inv_camera_pose: CameraPoseSE3,
        eps: float = 0.001
) -> ErrorSe3PoseJacobian:
    """ Estimate the Jacobian of pose to reprojection error function.
    We figure out the differential, which maps small change in one of 6 camera pose degrees of freedom
    to how much the error along 2 dimensions changes.
    """
    def err_eval(inv_camera_pose: CameraPoseSE3) -> ReprojectionErrorVector:
        return _compute_reprojection_error(point_3d, point_2d, inv_camera_pose)

    dfs = []

    for i in range(6):
        dx = np.zeros(6)
        dx[i] -= eps
        pre_camera_pose = SE3Matrix.exp(dx).as_matrix() @ inv_camera_pose

        dx = np.zeros(6)
        dx[i] += eps
        post_camera_pose = SE3Matrix.exp(dx).as_matrix() @ inv_camera_pose

        pre_error = err_eval(pre_camera_pose)
        post_error = err_eval(post_camera_pose)

        df = (post_error - pre_error) / eps / 2.
        dfs.append(df)

    J = np.array(dfs)

    return J


def estimate_J_analytically(
        point_3d: CamFlippedWorldCoords3D,
        inv_camera_pose: CameraPoseSE3
) -> ErrorSe3PoseJacobian:
    """

    """
    pc = inv_camera_pose @ point_3d
    inv_z = 1. / pc[2]
    inv_z2 = inv_z * inv_z

    J = np.array(([
        [-inv_z, 0],
        [0, -inv_z],
        [pc[0] * inv_z2, pc[1] * inv_z2],
        [pc[0] * pc[1] * inv_z2, 1 + pc[1] * pc[1] * inv_z2],
        [-1 - pc[0] * pc[0] * inv_z2, -pc[0] * pc[1] * inv_z2],
        [pc[1] * inv_z, -pc[0] * inv_z]
    ]))

    return J


def gauss_netwon_pnp(
    inverse_of_camera_pose_initial_guess: Optional[CameraPoseSE3],     # it  has to be in cam flipped keyframe !!
    points_3d_in_flipped_keyframe: CamFlippedWorldCoords3D,
    points_2d_in_img: ImgCoords2d,
    iterations: int = 10,   # convergence is quadratic, so 10 should be plenty
    verbose: bool = False
):
    inv_camera_pose = inverse_of_camera_pose_initial_guess

    for i in range(iterations):
        errs = []

        H = np.zeros((6, 6))
        b = np.zeros(6)

        # this for loop is a bit naive, but it's fast enough, so I'm not going to touch it
        for point_2d, point_3d in zip(points_2d_in_img, points_3d_in_flipped_keyframe):
            J = estimate_J_analytically(point_3d, inv_camera_pose)
            e = _compute_reprojection_error(point_3d, point_2d, inv_camera_pose)

            H += J @ J.T
            b += -J @ e

            errs.append(e)

        dx = np.linalg.solve(H, b)
        loss = np.linalg.norm(np.array(errs), axis=1).mean()
        inv_camera_pose = SE3Matrix.exp(dx).as_matrix() @ inv_camera_pose
        if verbose:
            print(f"i = {i} mse = {loss:.2f} dx = {dx.round(2)}")

    if verbose:
        print(f'Final reprojection MSE = {loss}')

    return inv_camera_pose


def _compute_reprojection_error(
        point_3d: CamFlippedWorldCoords3D,
        point_2d: ImgCoords2d,
        inv_camera_pose: CameraPoseSE3
) -> ReprojectionErrorVector:
    """ Compute the reprojection error for a single point."""
    pc = inv_camera_pose @ point_3d
    proj = pc[0] / pc[2], pc[1] / pc[2]
    e = point_2d - proj
    return e