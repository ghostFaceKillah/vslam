import numpy as np

from utils.custom_types import Array
from vslam.transforms import Vector3d


def vec_hat(x: Vector3d) -> Array['3,3', np.float64]:
    return np.array([
        [  0.,  -x[2],  x[1]],
        [ x[2],    0., -x[0]],
        [-x[1],  x[0],    0.]
    ], dtype=np.float64)
