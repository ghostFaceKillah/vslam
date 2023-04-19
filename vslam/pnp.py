from typing import Optional, Tuple

import attr
import numpy as np

from liegroups.numpy.se3 import SE3Matrix
from utils.custom_types import Array
from vslam.poses import correct_SE3_matrix_inplace
from vslam.transforms import SE3_inverse, WORLD_TO_CAM_FLIP
from vslam.types import CamFlippedWorldCoords3D, ImgCoords2d, CameraPoseSE3, ReprojectionErrorVector, WorldCoords3D, \
    TransformSE3

ErrorSe3PoseJacobian = Array['6,2', np.float64]


def estimate_J_numerically(
        point_3d: CamFlippedWorldCoords3D,
        point_2d: ImgCoords2d,
        camera_pose: CameraPoseSE3,
        eps: float = 0.001
) -> ErrorSe3PoseJacobian:
    """ Estimate the Jacobian of pose to reprojection error function.
    We figure out the differential, which maps small change in one of 6 camera pose degrees of freedom
    to how much the error along 2 dimensions changes.
    """

    def err_eval(camera_pose: CameraPoseSE3) -> ReprojectionErrorVector:
        return _compute_reprojection_error(point_3d, point_2d, camera_pose)

    dfs = []

    for i in range(6):
        dx = np.zeros(6)
        dx[i] -= eps
        pre_camera_pose = camera_pose @ SE3Matrix.exp(dx).as_matrix()

        dx = np.zeros(6)
        dx[i] += eps
        post_camera_pose = camera_pose @ SE3Matrix.exp(dx).as_matrix()

        pre_error = err_eval(pre_camera_pose)
        post_error = err_eval(post_camera_pose)

        df = (post_error - pre_error) / eps / 2.
        dfs.append(df)

    J = np.array(dfs)

    return J


def estimate_J_analytically(
        point_3d: CamFlippedWorldCoords3D,
        camera_pose: CameraPoseSE3
) -> ErrorSe3PoseJacobian:
    """

    """
    pc = WORLD_TO_CAM_FLIP @ SE3_inverse(camera_pose) @ point_3d
    inv_z = 1. / pc[2]
    inv_z2 = inv_z * inv_z

    # J = np.array(([
    #     [-pc[0] * inv_z2, -pc[1] * inv_z2],
    #     [inv_z, 0],
    #     [0, -inv_z],
    #     [pc[1] * inv_z, -pc[0] * inv_z],
    #     [pc[0] * pc[1] * inv_z2, 1 + pc[1] * pc[1] * inv_z2],
    #     [1 + pc[0] * pc[0] * inv_z2, pc[0] * pc[1] * inv_z2],
    # ]))

    J = np.array(([
        [-pc[0] * inv_z2, -pc[1] * inv_z2],
        [inv_z, 0],
        [1 + pc[0] * pc[0] * inv_z2, pc[0] * pc[1] * inv_z2],
    ]))

    return J


@attr.s(auto_attribs=True)
class GaussNetwonAuxillaryInfo:
    euclidean_errors: Array['N', np.float64]
    mean_euclidean_error: float


def gauss_netwon_pnp(
    camera_pose_initial_guess_in_keyframe: Optional[CameraPoseSE3],     # initial guess has to be relative to keyframe!
    points_3d_in_keyframe: WorldCoords3D,   # if those are in keyframe, the pose estimate will be relative to keyframe
    points_2d_in_img: ImgCoords2d,
    iterations: int = 20,   # convergence is quadratic, so 10 should be plenty
    verbose: bool = False
) -> Tuple[TransformSE3, GaussNetwonAuxillaryInfo]:
    camera_pose = camera_pose_initial_guess_in_keyframe

    for i in range(iterations):
        errs = []

        H = np.zeros((3, 3))
        b = np.zeros(3)

        # this for loop is a bit naive, but it's fast enough, so I'm not going to touch it
        for point_2d, point_3d in zip(points_2d_in_img, points_3d_in_keyframe):
            # J_num = estimate_J_numerically(point_3d, point_2d, camera_pose)
            J = estimate_J_analytically(point_3d, camera_pose)
            e = _compute_reprojection_error(point_3d, point_2d, camera_pose)

            H += J @ J.T
            b += -J @ e

            errs.append(e)

        dx = np.linalg.solve(H, b)
        errs = np.array(errs)   # reprojection error per axis
        euc_errs = np.linalg.norm(errs, axis=1)   # how much off on both axes
        loss = euc_errs.mean()
        real_dx = np.array([dx[0], dx[1], 0, 0, 0, dx[2]])
        camera_pose = correct_SE3_matrix_inplace(camera_pose @ SE3Matrix.exp(real_dx).as_matrix())
        if verbose:
            print(f"i = {i} mse = {loss:.2f} dx = {dx.round(2)}")

    if verbose:
        print(f'Final reprojection MSE = {loss}')

    aux_info = GaussNetwonAuxillaryInfo(euclidean_errors=euc_errs, mean_euclidean_error=loss)

    return camera_pose, aux_info


def _compute_reprojection_error(
        point_3d: CamFlippedWorldCoords3D,
        point_2d: ImgCoords2d,
        camera_pose: CameraPoseSE3
) -> ReprojectionErrorVector:
    """ Compute the reprojection error for a single point."""
    pc = WORLD_TO_CAM_FLIP @ SE3_inverse(camera_pose) @ point_3d
    proj = pc[0] / pc[2], pc[1] / pc[2]
    e = point_2d - proj
    return e