import attr
from jax import numpy as np

from vslam.poses import get_SE3_pose
from vslam.types import TransformSE3


@attr.s(auto_attribs=True)
class UiRequestedTransforms:
    camera: TransformSE3 = attr.Factory(get_SE3_pose)
    scene: TransformSE3 = attr.Factory(get_SE3_pose)

    @classmethod
    def empty(cls):
        return cls()


def key_to_maybe_transforms(key: int) -> UiRequestedTransforms:
    """ Transform key pressed by the user into camera and scene transformations """
    if key == -1:  # None
        return UiRequestedTransforms.empty()
    elif key == 0:   # up arrow key
        return UiRequestedTransforms(scene=get_SE3_pose(pitch=np.deg2rad(10)))
    elif key == 1:   # down arrow key
        return UiRequestedTransforms(scene=get_SE3_pose(pitch=np.deg2rad(-10)))
    elif key == 2:   # left arrow key
        return UiRequestedTransforms(scene=get_SE3_pose(roll=np.deg2rad(10)))
    elif key == 3:   # right arrow key
        return UiRequestedTransforms(scene=get_SE3_pose(roll=np.deg2rad(-10)))
    elif key == ord('w'):
        return UiRequestedTransforms(camera=get_SE3_pose(x=0.1))
    elif key == ord('s'):
        return UiRequestedTransforms(camera=get_SE3_pose(x=-0.1))
    elif key == ord('q'):
        return UiRequestedTransforms(camera=get_SE3_pose(y=-0.1))
    elif key == ord('e'):
        return UiRequestedTransforms(camera=get_SE3_pose(y=0.1))
    elif key == ord('a'):
        return UiRequestedTransforms(camera=get_SE3_pose(yaw=np.deg2rad(-2)))
    elif key == ord('d'):
        return UiRequestedTransforms(camera=get_SE3_pose(yaw=np.deg2rad(2)))
    else:
        print(f"Unknown keypress {key} {chr(key)}")
        return UiRequestedTransforms.empty()