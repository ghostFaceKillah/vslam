import numpy as np
from vslam.types import CameraRotationSO3, TransformSE3


def identity_rotation() -> CameraRotationSO3:
    return np.eye(3, dtype=np.float64)


def get_SO3_rotation_from_euler(
        yaw: float,
        pitch: float,
        roll: float
) -> CameraRotationSO3:
    yaw_matrix = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitch_matrix = np.matrix([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    roll_matrix = np.matrix([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # TODO: numerical error accumulation, very suspicious way to do it
    return yaw_matrix * pitch_matrix * roll_matrix


def get_SE3_pose(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0
) -> TransformSE3:
    pose = np.eye(4, dtype=np.float64)
    pose[0:3, 0:3] = get_SO3_rotation_from_euler(yaw, pitch, roll)
    pose[0:3, 3] = np.array([x, y, z], dtype=np.float64)

    return pose
