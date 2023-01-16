import attr
from jax import numpy as np

from utils.colors import BGRCuteColors
from utils.custom_types import Array, BGRColor
from vslam.types import TransformSE3


@attr.define
class RenderTriangle3d:
    """ a triangle floating in space """
    points: Array['3,4', np.float64]   # three 3d points
    front_face_color: BGRColor = BGRCuteColors.GRASS_GREEN
    back_face_color: BGRColor = BGRCuteColors.CRIMSON

    def mutate(self, transform: TransformSE3) -> 'RenderTriangle3d':
        result = transform @ self.points.T
        return RenderTriangle3d(
            points=result.T,
            front_face_color=self.front_face_color,
            back_face_color=self.back_face_color,
        )


RenderTrianglesPointsInCam = Array['N,3,4', np.float64]