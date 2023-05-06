from typing import Tuple

import numpy as np
from utils.custom_types import Array


Vector3d = Array['3', np.float64]
Point2d = Array['2', np.float64]
Line2d = Tuple[Point2d, Point2d]
Points2d = Array['N,2', np.float64]
TransformSE3 = Array['4,4', np.float64]
Pose2DArray = Array['3', np.float64]   # x, y, theta in a plane
CameraRotationSO3 = Array['3,3', np.float64]
ReprojectionErrorVector = Array['2', np.float64]

ArrayOfColors = Array['N,3', np.uint8]

WorldCoords3D = Array['N,3', np.float64]      # x away from us, y to right, z up, starts at world origin
CamFlippedWorldCoords3D = Array['N,4', np.float64]   # x right, y down, z out, starts at world origin, world coords flipped to cam coords
CamCoords4d = Array['N,4', np.float64]        # same as below, but x y z (1 if point 0 if direction)
CamCoords3d = Array['N,3', np.float64]        # x goes right, y down, z out, looking out of cam, origin is optical center
CamCoords3dHomog = Array['N,3', np.float64]   # X/Z Y/Z 1
ImgCoords2d = Array['N,2', np.float64]        # x goes right, y down, homogenous without z,
PxCoords2d = Array['N,2', np.int64]           # int, x goes down, y goes right, divided by fx/fy, subtracted cx, cy
                                              # please notice that cv pixel coordinates are swapped

"""
WorldCoords3D
    -[coordinate flip, x = y, y = -z, z = x] -> 
        CamFlippedWorldCoords3D
            -[inv(CameraPoseSE3) * point coords]->
                CamCoords3d
                    -[/Z] -> 
                        CamCoords3dHomog 
                            -[undistortion]->
                                    ImgCoords3d
                                        -[drop third dim]->
                                            ImgCoords2d
                                                -[-center, *px scale and focal (intrinsics), flip (!)]
                                                    -> PxCoords2d
"""
OpenCVPixel = Tuple[int, int]   # right, down, non-negative
# images
BGRImageArray = Array['H,W,3', np.uint8]

