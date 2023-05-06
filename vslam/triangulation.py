from typing import List, Optional

import attr
import numpy as np

from vslam.math import vec_hat
from vslam.types import TransformSE3, CamCoords3dHomog


def _docs_of_naive_triangulation() -> str:
    """
    Function naive_triangulation implements Triangulation based on
     change-of-coordinate system equations.

    It takes as arguments:
     - coordinates of sequence of points in frames (coordinates) of two cameras
     - transform from cam one to cam two (aka pose of cam 1 in frame of cam 2)
    It outputs sequence of depths or None if estimate is too wrong.

    It is "naive" in the sense that it doesn't treat sources of error in
    the estimate in elegant ways.

    We operate in the frame (coordinate system) of cam one.
    We compare coordinates of the same sequence of points.
        - points, as seen in cam one
        - points as seen in cam two, then fransformed again to cam two

    If there were no errors (point matching was perfect, pose transform
    was perfect, etc, etc), these would be equal and we could solve for depth.
    In practice, there is noise, so we can only hope minimize error.

    Let us take rotation and translation [R, t] from cam_one_in_two. Then:

     s_2  x_2 = s_1 R x_1 + t
     multiply both sides by x_2^hat (wiki skew vector)
     s_2 x_2^hat x_2 = 0 = s_1 x_2^hat R x_1 + x2^hat t
                           s * | - - a - - |   |- b - |
     and in this way we have equation of type s * a + b = 0
     where a, b are 3d vectors, and s is scale scalar.
     In reality, a, b have only "2 dimensions of freedom" (exercise for the reader)
     We can solve for s two times and compare the results ("minimize the error").
     If the results are close, we have good depth estimate.

     For details, refer to 'Introduction to VSLAM', 6.5 Triangulation, for the details.
     """


@attr.define
class DepthEstimate:
    depth_or_none: Optional[float]
    depth_est_std: float


def naive_triangulation(
    points_in_cam_one: CamCoords3dHomog,
    points_in_cam_two: CamCoords3dHomog,
    cam_one_in_two: TransformSE3
) -> List[DepthEstimate]:
    """ Estimate depth based on coordinate transform formula. See above for longer doc.
    Returns None if there seems to be too much error in inputs. """

    R = cam_one_in_two[:3, :3]
    t = cam_one_in_two[:3, 3]

    scales = []

    for i in range(len(points_in_cam_one)):   # this is also naive! loops in python are evil
        x1 = points_in_cam_one[i]
        x2 = points_in_cam_two[i]

        x2_hat = vec_hat(x2)

        a = x2_hat @ R @ x1
        b = x2_hat @ t
        # s * a + b = 0
        # s = - b / a
        num_stable_flag = (np.abs(a) > 0.01) & (np.abs(b) > 0.01)   # this is the naive part

        a = a[num_stable_flag]
        b = b[num_stable_flag]

        s = -b / a

        # further naive success filtering
        if s.std() < 0.01 and s.mean() > 0:
            scales.append(DepthEstimate(s.mean(), s.std()))
        else:
            scales.append(DepthEstimate(None, s.std()))

    return scales
