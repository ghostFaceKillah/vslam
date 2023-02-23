from typing import List, Optional

import numpy as np

from vslam.math import vec_hat
from vslam.types import CameraPoseSE3, CamCoords3dHomog


def naive_triangulation(
    points_in_cam_one: CamCoords3dHomog,
    points_in_cam_two: CamCoords3dHomog,
    cam_two_in_cam_one: CameraPoseSE3     # TODO: Make sure it's like that, I think it should be opposite ?
) -> List[Optional[float]]:
    """ We assume T1 to be identity (we compute everything in image frame of camera 1) """

    R = cam_two_in_cam_one[:3, :3]
    t = cam_two_in_cam_one[:3, 3]

    scales = []

    for i in range(len(points_in_cam_one)):
        x1 = points_in_cam_one[i]
        x2 = points_in_cam_two[i]

        x2_hat = vec_hat(x2)

        a = x2_hat @ R @ x1
        b = x2_hat @ t
        # s * a + b = 0
        # s = - b / a
        num_stable_flag = (np.abs(a) > 0.01) & (np.abs(b) > 0.01)

        a = a[num_stable_flag]
        b = b[num_stable_flag]

        s = -b / a

        # dirty success filtering
        if s.std() < 0.5 and s.mean() > 0:
            scales.append(s.mean())
        else:
            scales.append(None)

    return scales
