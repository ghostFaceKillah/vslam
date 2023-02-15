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
from vslam.poses import get_SE3_pose
from vslam.transforms import get_world_to_cam_coord_flip_matrix, SE3_inverse


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

    camera_pose = get_SE3_pose(x=-3.5)
    second_camera_pose = get_SE3_pose(x=-3.0)

    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()
    cam_pose_inv = SE3_inverse(camera_pose)
    points = onp.concatenate([tri.points for tri in triangles])
    points_in_cam_one = points @  SE3_inverse(camera_pose).T @ world_to_cam_flip.T

    points_in_cam_two = points @ SE3_inverse(second_camera_pose).T @ world_to_cam_flip.T

    unit_depth_cam_points = points_in_cam_two[..., :-1]
    triangle_depths = unit_depth_cam_points[..., -1]
    triangles_in_img_coords = (unit_depth_cam_points / triangle_depths[..., onp.newaxis])[..., :-1]
    # ws = (np.arange(0, screen_w) - cam_intrinsics.cx) / cam_intrinsics.fx
    # hs = (np.arange(0, screen_h) - cam_intrinsics.cy) / cam_intrinsics.fy
    cx = camera_specs.cam_intrinsics.cx
    cy = camera_specs.cam_intrinsics.cy
    fx = camera_specs.cam_intrinsics.fx
    fy = camera_specs.cam_intrinsics.fy

    # triangles_in_img_coords = (triangles_in_img_coords - onp.array([cx, cy])) / onp.array([fx, fy])

    return camera_pose, points_in_cam_one, triangles_in_img_coords


if __name__ == '__main__':
    camera_pose, points_3d, points_2d = get_data()

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
        # we need to be very careful about the dimensions - there is difference between book and sophus code
        # we need to have good intuitive understanding of variables

    # should I look for NaNs ?

    """
    dx = H.ldlt().solve(b);

    if (iter > 0 && cost >= lastCost) {
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;
    """
