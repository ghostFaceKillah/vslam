

"""

/Users/misiu-dev/src/robots/build/pose_estimation_2d2d
*/

"""

import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional




def pixel_2_cam(
        px: float, py: float, fx: float, fy: float, cx: float, cy: float
):
    return (
        (px - cx) / fx,
        (py - cy) / fy
    )


def cam_2_pixel(x: float, y: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[int, int]:
    px = int(fx * x + cx)
    py = int(fy * y + cy)
    return px, py


def depth_to_color(depth: float) -> Tuple[int]:
    unit_depth = (np.clip(depth, 5.0, 20.0) - 5.0)/ (20. - 5.)
    b = int(unit_depth * 255.)
    g = 0
    r = int((1 - unit_depth) * 255.)
    return b, g, r


# Initialize consts to be used in linear_LS_triangulation()
linear_LS_triangulation_C = -np.eye(2, 3)


def linear_LS_triangulation(u1, P1, u2, P2):
    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

    The status-vector will be True for all points.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # Create array of triangulated points
    x = np.zeros((3, len(u1)))

    # Initialize C matrices
    C1 = np.array(linear_LS_triangulation_C)
    C2 = np.array(linear_LS_triangulation_C)

    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]

        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
        b *= -1

        # Solve for x vector
        cv2.solve(A, b, x[:, i:i + 1], cv2.DECOMP_SVD)

    return x.T, np.ones(len(u1), dtype=bool)


def vec_hat(x):
    return np.array([
        [  0.,  -x[2],  x[1]],
        [ x[2],    0., -x[0]],
        [-x[1], x[0],    0.]
    ])
    pass


def mike_triangulation(pts_1, T1, pts_2, T2) -> List[Optional[float]]:

    R = T2[:, :3]
    t = T2[:, 3]

    scales = []
    pts_3d = []

    for i in range(len(pts_1)):
        x1 = np.array([pts_1[i, 0], pts_1[i, 1], 1.0])
        x2 = np.array([pts_2[i, 0], pts_2[i, 1], 1.0])
        x2_hat = vec_hat(x2)

        a = x2_hat @ R @ x1
        b = x2_hat @ t
        # s * a + b = 0
        # s = - b / a
        s = - b / a

        if s.std() < 0.5 and s.mean() > 0:
            scales.append(s.mean())
        else:
            scales.append(None)

    return scales



if __name__ == '__main__':
    # img_2_path = "/Users/misiu-dev/temp/phd/2022/mav0/cam0/data/1403636582763555584.png"
    # img_1_path = "/Users/misiu-dev/temp/phd/2022/mav0/cam0/data/1403636583013555456.png"
    # # EUROC
    # fx = 458.654
    # fy = 457.296
    # cx = 367.215
    # cy = 248.375

    # super doesn't work
    img_1_path = "/Users/misiu-dev/temp/phd/kitti-dataset/sequences/00/image_0/000000.png"
    img_2_path = "/Users/misiu-dev/temp/phd/kitti-dataset/sequences/00/image_1/000000.png"

    fx = 517.3
    fy = 516.5
    cx = 325.1
    cy = 249.7


    focal_len = fx

    img_1 = cv2.imread(img_1_path)
    img_2 = cv2.imread(img_2_path)

    orb = cv2.ORB_create(100000)

    img_1_kp, img_1_desc = orb.detectAndCompute(img_1, None)
    img_2_kp, img_2_desc = orb.detectAndCompute(img_2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(img_1_desc, img_2_desc)

    # Show the final image
    # final_img = cv2.drawMatches(img_1, img_1_kp, img_2, img_2_kp, matches, None)
    # cv2.imshow("Matches", final_img)
    # cv2.waitKey(-1)

    min_dist = min([m.distance for m in matches])
    max_dist = max([m.distance for m in matches])
    print(f"{min_dist=} {max_dist=}")

    pts1 = []
    pts2 = []
    for i, m in enumerate(matches):
        if m.distance < max(2 * min_dist, 30.0):
            pts1.append(img_1_kp[m.queryIdx].pt)
            pts2.append(img_2_kp[m.trainIdx].pt)

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy))

    retVal, R, t, mask = cv2.recoverPose(E, pts1, pts2)

    T1 = np.hstack([np.eye(3), np.zeros(shape=(3,1))])
    T2 = np.hstack([R, t])

    def transform(pts):
        cam_pts = []
        for pt in pts:
            px, py = pt
            cam_x, cam_y = pixel_2_cam(px, py, fx, fy, cx, cy)
            cam_pts.append((cam_x, cam_y))
        return np.asarray(cam_pts)

    cam_pts_1 = transform(pts1)
    cam_pts_2 = transform(pts2)


    scales = mike_triangulation(cam_pts_1, T1, cam_pts_2, T2)
    # pts_3d_2, status = linear_LS_triangulation(cam_pts_1, T1, cam_pts_2, T2)

    img_to_draw = img_1.copy()
    for i, pt in enumerate(pts1):
        z = scales[i]
        if z is not None:
            px, py = pt
            color = depth_to_color(float(z))

            cv2.circle(img_to_draw, (int(px), int(py)), 2, color, -1)

    cv2.imshow("depths", img_to_draw)
    cv2.waitKey(-1)



    print(F)
    print(mask)

