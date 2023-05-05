from typing import Optional

import attr
import numpy as np
from scipy.spatial.transform import Rotation as R

from vslam.types import CameraPoseSE3


@attr.define
class VelocityPoseTracker:
    # it predicts the next pose
    current_pose_estimate: CameraPoseSE3
    last_pose_estimate: Optional[CameraPoseSE3] = None
    time_delta: float = 1.0

    @classmethod
    def from_defaults(cls):
        return cls(current_pose_estimate=np.eye(4, np.float64))

    def track(self, new_pose: CameraPoseSE3):
        self.last_pose_estimate = self.current_pose_estimate
        self.current_pose_estimate = new_pose

    def get_next_baselink_in_world_pose_estimate(self) -> CameraPoseSE3:
        if self.last_pose_estimate is None:
            return self.current_pose_estimate

        # Compute linear velocity
        linear_velocity = (self.current_pose_estimate[:3, 3] - self.last_pose_estimate[:3, 3]) / self.time_delta

        # Compute angular velocity
        current_rot = R.from_matrix(np.copy(self.current_pose_estimate[:3, :3]))
        last_rot = R.from_matrix(np.copy(self.last_pose_estimate[:3, :3]))
        relative_rot = last_rot.inv() * current_rot
        angular_velocity = relative_rot.as_rotvec() / self.time_delta

        # Predict next pose
        next_translation = self.current_pose_estimate[:3, 3] + linear_velocity * self.time_delta
        next_rotation = (R.from_rotvec(angular_velocity * self.time_delta) * current_rot).as_matrix()

        next_pose = np.eye(4, dtype=np.float64)
        next_pose[:3, :3] = next_rotation
        next_pose[:3, 3] = next_translation

        return next_pose
