"""
We are going to learn go PnP = perspective and point.
Algorithm that computes pose differnce based on matching between 3d point cloud to 2d oints

We are going to take 3 pictures
- left eye, right eye
- next frame from left eye

we are going to match 3d points and 2d points
by minimizing reprojection error using manual Gauss-Newton implementation.

see this https://github.com/gaoxiang12/slambook2/blob/master/ch7/pose_estimation_3d2d.cpp#L172
bundleAdjustmentGaussNewton

6.8.2. Pose Estimation from Scratch
and
6.7.3 Solve PnP by minimizing the reprojection error, page 177
"""
import cv2
import numpy as np
import numpy as onp

from liegroups.numpy.se3 import SE3Matrix
from sim.actor_simulation import TriangleSceneRenderer
from sim.sample_scenes import get_two_triangle_scene
from sim.sim_types import CameraSpecs
from utils.profiling import just_time
from vslam.poses import get_SE3_pose
from vslam.transforms import SE3_inverse, get_world_to_cam_coord_flip_matrix


def check_intuition_about_data():
    """
    we need to get 3d points and 2d points with matches between them.

    Let's take a scene and compute points based on that
    """
    renderer = TriangleSceneRenderer.from_easy_scene()
    camera_pose = get_SE3_pose(x=-3.5)
    img = renderer.render_first_person_view(camera_pose)

    cv2.imshow('scene', img)
    key = cv2.waitKey(-1)

    renderer = TriangleSceneRenderer.from_easy_scene()
    camera_pose = get_SE3_pose(x=-2.0)
    img = renderer.render_first_person_view(camera_pose)

    cv2.imshow('scene', img)
    key = cv2.waitKey(-1)


def get_data():
    triangles = get_two_triangle_scene()
    camera_specs = CameraSpecs.from_default()

    camera_pose = get_SE3_pose(z=-3.5)
    second_camera_pose = get_SE3_pose(z=-3.0)

    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()


    # cam_pose_inv = SE3_inverse(camera_pose)
    points = onp.concatenate([tri.points for tri in triangles])
    points = points @ world_to_cam_flip.T

    points_in_cam_one = points @  SE3_inverse(camera_pose).T
    points_in_cam_two = points @ SE3_inverse(second_camera_pose).T

    unit_depth_cam_points = points_in_cam_two[..., :-1]
    triangle_depths = unit_depth_cam_points[..., -1]
    triangles_in_img_coords = (unit_depth_cam_points / triangle_depths[..., onp.newaxis])[..., :-1]

    return SE3_inverse(camera_pose), points, triangles_in_img_coords


def ok():
    camera_pose, points_3d, points_2d = get_data()

    """
    ideas:

    1) what is the interpretation of the axes
    """

    # one iteration, do this for many iterations
    camera_specs = CameraSpecs.from_default()
    cx = camera_specs.cam_intrinsics.cx
    cy = camera_specs.cam_intrinsics.cy
    fx = camera_specs.cam_intrinsics.fx
    fy = camera_specs.cam_intrinsics.fy

    # cam flip can be the trouble here ...

    for it in range(100):
        cost = 0.0
        H = np.zeros((6, 6))
        b = np.zeros((1, 6))

        for point_2d, point_3d in zip(points_2d, points_3d):
            pc = camera_pose @ point_3d
            inv_z = 1. / pc[2]
            inv_z2 = inv_z * inv_z

            # proj = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
            proj = pc[0] / pc[2], pc[1] / pc[2]
            e = point_2d - proj

            cost += np.sqrt(e @ e)

            J = np.array([
                [-inv_z, 0],
                [pc[0] * inv_z2, pc[0] * pc[1] * inv_z2],
                [-1 - 1 * pc[0] * pc[0] * inv_z2, pc[1] * inv_z],
                [0, -inv_z],
                [pc[1] * inv_z2, 1 + pc[1] * pc[1] * inv_z2],
                [-pc[0] * pc[1] * inv_z2, -pc[0] * inv_z]
            ])

            H += J @ J.T
            b -= J @ e

        dx = np.linalg.solve(H, b[0])
        diff = SE3Matrix.exp(dx).as_matrix()
        camera_pose = diff @ camera_pose

        print(f"dx={dx.round(4)}")
        print(f"{cost=:.2f}")

        a = 1


def what_is_meaning_of_the_axes():
    camera_pose, points_3d, points_2d = get_data()
    print(camera_pose)

    for i in range(6):
        dx = np.zeros(6)
        dx[i] += 0.1
        diff = SE3Matrix.exp(dx).as_matrix()
        post_camera_pose = diff @ camera_pose

        print(f"{i=}")
        print(post_camera_pose)
        print(" ")

    # it is literally the simplest thing
    # pose: x, y, z
    # angle: spin around x, y, z axes


def compute_reprojection_error(point_3d, point_2d, inv_camera_pose):
    pc = inv_camera_pose @ point_3d
    proj = pc[0] / pc[2], pc[1] / pc[2]
    e = point_2d - proj
    return e


def estimate_J_numerically(point_3d, point_2d, inv_camera_pose):
    def err_eval(inv_camera_pose):
        return compute_reprojection_error(point_3d, point_2d, inv_camera_pose)

    eps = 0.001

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


def estimate_J_analytically(point_3d, point_2d, inv_camera_pose):
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


def overfit_one_point():
    inv_camera_pose, points_3d, points_2d = get_data()

    # major mess in coordinates again - we are dividing by Z, which in our case is world height

    point_3d = points_3d[0]
    point_2d = points_2d[0]

    for i in range(100):
        J = estimate_J_numerically(point_3d, point_2d, inv_camera_pose)
        e = compute_reprojection_error(point_3d, point_2d, inv_camera_pose)
        dx = J @ e
        inv_camera_pose = SE3Matrix.exp(-0.01 * dx).as_matrix() @ inv_camera_pose
        loss = np.sqrt(e @ e)

        print(f"i = {i} loss = {loss}")


def solve_points_first_order(verbose: bool = False):
    """ Naive method: numerical derivative, direct gradient """
    inv_camera_pose, points_3d, points_2d = get_data()

    for i in range(100):
        Js = []
        errs = []
        dxs = []
        jac_err = []

        for point_2d, point_3d in zip(points_2d, points_3d):
            J = estimate_J_numerically(point_3d, point_2d, inv_camera_pose)
            J2 = estimate_J_analytically(point_3d, point_2d, inv_camera_pose)

            e = compute_reprojection_error(point_3d, point_2d, inv_camera_pose)
            dx = J2 @ e
            jac_err.append(np.abs(J - J2).sum())
            Js.append(J)
            errs.append(e)
            dxs.append(dx)

        # that's where you do gauss newton yo
        dx_est = np.array(dxs).mean(axis=0)
        mean_abs_jac_err = np.array(jac_err).mean()

        loss = np.linalg.norm(np.array(errs), axis=1).mean()
        inv_camera_pose = SE3Matrix.exp(-1 * dx_est).as_matrix() @ inv_camera_pose

        if verbose:
            print(f"i = {i} mse = {loss:.2f} dx = {dx_est.round(2)} jac_err = {mean_abs_jac_err:.5f}")

    print("final result")
    print(inv_camera_pose)



if __name__ == '__main__':
    # what_is_meaning_of_the_axes()
    # overfit_one_point()
    with just_time():
        solve_points_first_order()
    # I can do it numerically for fun
    # obvious bug - reprojecting the native camera results in error, wtf


