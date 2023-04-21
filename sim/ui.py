import attr
from jax import numpy as np

from vslam.poses import get_SE3_pose
from vslam.types import TransformSE3


@attr.s(auto_attribs=True)
class InteractionTransforms:
    camera: TransformSE3 = attr.Factory(get_SE3_pose)
    scene: TransformSE3 = attr.Factory(get_SE3_pose)

    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def go_straight(cls):
        return cls(camera=get_SE3_pose(x=0.1))

    @classmethod
    def go_back(cls):
        return cls(camera=get_SE3_pose(x=-0.1))

    @classmethod
    def turn_right(cls):
        return cls(camera=get_SE3_pose(x=0.05, yaw=np.deg2rad(1)))

    @classmethod
    def turn_left(cls):
        return cls(camera=get_SE3_pose(x=0.05, yaw=np.deg2rad(-1)))


def key_to_maybe_transforms(key: int) -> InteractionTransforms:
    """ Transform key pressed by the user into camera and scene transformations """
    if key == -1:  # None
        return InteractionTransforms.empty()
    elif key == 0:   # up arrow key
        return InteractionTransforms(scene=get_SE3_pose(pitch=np.deg2rad(10)))
    elif key == 1:   # down arrow key
        return InteractionTransforms(scene=get_SE3_pose(pitch=np.deg2rad(-10)))
    elif key == 2:   # left arrow key
        return InteractionTransforms(scene=get_SE3_pose(roll=np.deg2rad(10)))
    elif key == 3:   # right arrow key
        return InteractionTransforms(scene=get_SE3_pose(roll=np.deg2rad(-10)))
    elif key == ord('w'):
        return InteractionTransforms.go_straight()
    elif key == ord('s'):
        return InteractionTransforms.go_back()
    elif key == ord('q'):
        return InteractionTransforms(camera=get_SE3_pose(y=-0.1))
    elif key == ord('e'):
        return InteractionTransforms(camera=get_SE3_pose(y=0.1))
    elif key == ord('a'):
        return InteractionTransforms.turn_left()
    elif key == ord('d'):
        return InteractionTransforms.turn_right()
    else:
        print(f"Unknown keypress {key} {chr(key)}")
        return InteractionTransforms.empty()