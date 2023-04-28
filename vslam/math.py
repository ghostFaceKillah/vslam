import numpy as np

from utils.custom_types import Array
from vslam.types import Vector3d


def vec_hat(x: Vector3d) -> Array['3,3', np.float64]:
    return np.array([
        [  0.,  -x[2],  x[1]],
        [ x[2],    0., -x[0]],
        [-x[1],  x[0],    0.]
    ], dtype=np.float64)


def normalize_vector(x: Vector3d) -> Vector3d:
    return x / np.linalg.norm(x)


def dot_product(x: Vector3d, y: Vector3d) -> Vector3d:
    # probably need to defend stuff like zero norm
    return x @ y / np.linalg.norm(x) / np.linalg.norm(y)


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_difference_of_angles(theta_one: float, theta_two: float) -> float:
    return normalize_angle(theta_one - theta_two)
