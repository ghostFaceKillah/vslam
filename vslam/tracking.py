import attr
import numpy as np

from vslam.types import CameraPoseSE3


@attr.define
class VelocityPoseTracker:
    # it predicts the next pose
    current_pose_estimate: CameraPoseSE3

    @classmethod
    def from_defaults(cls):
        return cls(current_pose_estimate=np.eye(4, np.float64))

    def track(self, new_pose: CameraPoseSE3):
        # TODO: propagate it forward by one step
        self.current_pose_estimate = new_pose

    def get_next_baselink_in_world_pose_estimate(self) -> CameraPoseSE3:
        return self.current_pose_estimate
