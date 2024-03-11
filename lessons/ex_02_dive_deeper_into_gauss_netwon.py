"""
We are going to learn go PnP = perspective and point.
This algorithm computes pose of camera based on matching between 3d point cloud to 2d points on an image.

We take 3d point cloud from simulation and 2d points projected onto image plane.
We treat the pose as unknown.

It is a simplified setup that allows us to focus on the algorithm itself and completely debug it.
In reality:
- depth of a point (and thus its 3d position) is not known and has to be read from a noisy sensor or triangulated
  from stereo
- depth of a point is often treated as an optimization variable

see this https://github.com/gaoxiang12/slambook2/blob/master/ch7/pose_estimation_3d2d.cpp#L172
bundleAdjustmentGaussNewton

6.8.2. Pose Estimation from Scratch
and
6.7.3 Solve PnP by minimizing the reprojection error, page 177
"""

import attr
import numpy as np

from liegroups.numpy.se3 import SE3Matrix
from sim.sample_scenes import get_two_triangle_scene
from utils.profiling import just_time
from vslam.pnp import (
    estimate_J_numerically,
    estimate_J_analytically,
    gauss_netwon_pnp,
    _compute_reprojection_error,
)
from vslam.poses import get_SE3_pose
from vslam.transforms import SE3_inverse, get_world_to_cam_coord_flip_matrix
from vslam.types import CamFlippedWorldCoords3D, ImgCoords2d, TransformSE3


@attr.define
class _PoseEstimationData:
    camera_pose: TransformSE3
    camera_pose_initial_guess: TransformSE3
    keyframe_pose: TransformSE3
    points_3d_in_keyframe: CamFlippedWorldCoords3D
    points_2d_in_img: ImgCoords2d

    @classmethod
    def example(cls) -> "_PoseEstimationData":
        triangles = get_two_triangle_scene()

        # our initial, incorrect guess
        cam_pose_initial_guess = get_SE3_pose(x=-3.5)
        # the ground truth - we want to find this
        true_camera_pose = get_SE3_pose(x=-3.0, y=0.3, yaw=0.01)

        # we are comparing against points registered wrt this pose
        keyframe_pose = get_SE3_pose()

        world_to_cam_flip = get_world_to_cam_coord_flip_matrix()

        points_3d_in_world = np.concatenate([tri.points for tri in triangles])
        points_3d_in_keyframe = (SE3_inverse(keyframe_pose) @ points_3d_in_world.T).T

        points_in_cam_homogenous = (
            world_to_cam_flip @ SE3_inverse(true_camera_pose) @ points_3d_in_world.T
        ).T
        points_in_cam = points_in_cam_homogenous[..., :-1]
        point_depths = points_in_cam[..., -1]
        points_2d_in_img = (points_in_cam / point_depths[..., np.newaxis])[..., :-1]

        return cls(
            camera_pose=true_camera_pose,
            camera_pose_initial_guess=cam_pose_initial_guess,
            keyframe_pose=keyframe_pose,
            points_3d_in_keyframe=points_3d_in_keyframe,
            points_2d_in_img=points_2d_in_img,
        )


def _experiment_what_is_the_meaning_of_the_axes():
    """Check if `liegroups` has the same conventions around coordinates as we do."""

    data = _PoseEstimationData.example()

    print("baseline pose")
    print(data.keyframe_pose.round(2))

    for i in range(6):
        dx = np.zeros(6)
        dx[i] += 0.1
        diff = SE3Matrix.exp(dx).as_matrix()
        # should we multiply on the right on left ?
        # in the book, they did:     diff @ data.inverse_of_keyframe_pose
        post_camera_pose = diff @ data.keyframe_pose

        print(f"{i=}")
        print(post_camera_pose.round(2))
        print(" ")

    # it is as in the book / as in the "canonical" SE(3):
    # pose: (x, y, z) and then, angle: spin around x, y, z axes


def _overfit_one_point():
    """Make sure whether Jacobian is useful at all (doesn't contain obvious bugs).
    We consider a drastically simplified setting of minimizing reprojection error for one point only.
    Backpropagating through negated Jacobian should result in gradual lowering of error and convergence.
    """
    data = _PoseEstimationData.example()

    point_3d = data.points_3d_in_keyframe[0]
    point_2d = data.points_2d_in_img[0]
    camera_pose = data.camera_pose_initial_guess

    for i in range(100):
        J = estimate_J_numerically(point_3d, point_2d, camera_pose)
        e = _compute_reprojection_error(point_3d, point_2d, camera_pose)
        dx = J @ e
        camera_pose = camera_pose @ SE3Matrix.exp(-0.1 * dx).as_matrix()
        loss = np.sqrt(e @ e)

        print(f"i = {i} loss = {loss}")

    assert loss < 1e-6
    print("Final pose estimate is")
    print(camera_pose)
    print("Ground truth pose is ")
    print(SE3_inverse(data.inverse_of_camera_pose).round(2))
    print("Remember it's one point only so we can easily get incorrect solution")


def _solve_many_points_first_order_descent(verbose: bool = True):
    """Naive method: numerical derivative, direct gradient.
    On the way there, make sure that analytic and numeric jacobians are close to each other.
    """
    data = _PoseEstimationData.example()

    camera_pose = data.camera_pose_initial_guess
    points_2d = data.points_2d_in_img
    points_3d = data.points_3d_in_keyframe

    for i in range(4000):
        Js, errs, dxs, jac_err = [], [], [], []

        for point_2d, point_3d in zip(points_2d, points_3d):
            J_num = estimate_J_numerically(point_3d, point_2d, camera_pose)
            J = estimate_J_analytically(point_3d, camera_pose)

            e = _compute_reprojection_error(point_3d, point_2d, camera_pose)
            dx = J @ e
            jac_err.append(np.abs(J - J_num).sum())
            Js.append(J)
            errs.append(e)
            dxs.append(dx)

        dx_est = np.array(dxs).mean(axis=0)
        mean_abs_jac_err = np.array(jac_err).mean()
        loss = np.linalg.norm(np.array(errs), axis=1).mean()
        camera_pose = camera_pose @ SE3Matrix.exp(-dx_est).as_matrix()

        assert mean_abs_jac_err < 1e-4

        if verbose and i % 10 == 0:
            print(
                f"i = {i} mse = {loss:.2f} dx = {dx_est.round(2)} jac_err = {mean_abs_jac_err:.5f}"
            )

    print("Final pose estimate is")
    print(camera_pose.round(2))
    print("Ground truth pose is ")
    print(data.camera_pose.round(2))
    # So many iterations are needed to get to low error !
    # especially between 1k and 3k iterations the step size is tiny and error doesn't get low quickly enough
    # Second order helps a lot


def _solve_many_points_gauss_newton(verbose: bool = False):
    """Pose estimation that is somewhat close to production:
    1) Gauss newton second order method
    2) analytical gradient
    not vectrorized though!
    """

    data = _PoseEstimationData.example()

    camera_pose = gauss_netwon_pnp(
        data.camera_pose_initial_guess,
        data.points_3d_in_keyframe,
        data.points_2d_in_img,
        verbose=True,
    )

    print("Final pose estimate is")
    print(np.array(camera_pose).round(2))
    print("Ground truth pose is ")
    print(data.camera_pose.round(2))

    print("Final pose estimation error")
    print((camera_pose - data.camera_pose).round(2))


if __name__ == "__main__":
    # _experiment_what_is_the_meaning_of_the_axes()
    # _overfit_one_point()

    # with just_time():
    #     _solve_many_points_first_order_descent()

    with just_time():
        _solve_many_points_gauss_newton()

    # I can do it numerically for fun
    # obvious bug - reprojecting the native camera results in error, wtf
