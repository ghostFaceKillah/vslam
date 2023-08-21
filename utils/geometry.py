import dataclasses
from typing import Tuple

import numpy as np

from vslam.types import Point2d, Line2d


def cart_to_pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol_to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def wrap_angle(theta):
    """Cast angle to (-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def diff_angle(a1, a2):
    wrap_angle(a1 - a2)


@dataclasses.dataclass
class Arrow2d:
    """ This is in floating point coordinates. """
    # TODO: this should be in a separate `drawing` library file

    orientation: float
    origin_point: Point2d
    length: float

    arrowhead_length: float
    arrowhead_angle: float

    @classmethod
    def from_length_and_origin(cls, orientation: float, length: float, origin: Point2d) -> 'Arrow2d':

        return cls(
            orientation=orientation,
            origin_point=origin,
            length=length,
            arrowhead_length=0.2*length,
            arrowhead_angle=np.deg2rad(45.)
        )

    def get_from_pt(self) -> Point2d:
        return self.origin_point

    def get_to_pt(self) -> Point2d:
        return self.origin_point + pol_to_cart(self.length, self.orientation)

    def _get_arrowhead_left_arm_endpoint(self):
        return self.get_to_pt() - pol_to_cart(self.arrowhead_length, self.orientation - self.arrowhead_angle)

    def _get_arrowhead_right_arm_endpoint(self):
        return self.get_to_pt() - pol_to_cart(self.arrowhead_length, self.orientation + self.arrowhead_angle)

    def get_lines_to_draw(self) -> Tuple[Line2d, Line2d, Line2d]:
        start_point = self.origin_point
        end_point = self.get_to_pt()
        left_arrowhead_end_point = self._get_arrowhead_left_arm_endpoint()
        right_arrowhead_end_point = self._get_arrowhead_right_arm_endpoint()

        return (
            (start_point, end_point),   # principal line
            (end_point, left_arrowhead_end_point),    # left arrowhead line
            (end_point, right_arrowhead_end_point),    # right arrowhead line
        )
