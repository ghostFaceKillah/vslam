from typing import List, Optional

from vslam.math import vec_hat
from vslam.types import CameraPoseSE3, CamCoords3dHomog


def naive_triangulation(
    pts_1: CamCoords3dHomog,
    pts_2: CamCoords3dHomog,
    T2: CameraPoseSE3     # SE3 pose of the second camera in cam one frame
) -> List[Optional[float]]:
    """ We assume T1 to be identity (we compute everything in image frame of camera 1) """

    R = T2[:3, :3]
    t = T2[:3, 3]

    scales = []

    for i in range(len(pts_1)):
        x1 = pts_1[i]
        x2 = pts_2[i]

        x2_hat = vec_hat(x1)

        a = x2_hat @ R @ x1
        b = x2_hat @ t
        # s * a + b = 0
        # s = - b / a
        s = - b / a

        # very dirty success filtering
        if s.std() < 0.5 and s.mean() > 0:
            scales.append(s.mean())
        else:
            scales.append(None)

    return scales
