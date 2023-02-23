import attr
import numpy as np

from utils.custom_types import Array

Vector3d = Array['3', np.float64]
Point2d = Array['2', np.float64]
Points2d = Array['N,2', np.float64]

CameraIntrinsicMatrix = Array['3,3', np.float64]
Cam3dHomogToPxMatrix = Array['3,2', np.float64]


@attr.s(auto_attribs=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def to_matrix(self) -> CameraIntrinsicMatrix:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_homo_cam_coords_to_px_coords_matrix(self):
        # CamCoords3dHomog to PxCoords2d matrix
        return np.array([
            [0, self.fy, self.cy],
            [self.fx, 0, self.cx],
        ], dtype=np.float64)


"""
perception equation in the camera frame
---------------------------------------
  takes points directly from CamCoords3d to PxCoords2d
     [u]    [ f_x,   0, c_x]    [x] 
Z *  [v] =  [   0, f_y, c_y]  @ [y] = def = K @ P
     [1]    [   0,   0,   1]    [z]
     ^       ^
     :        Camera Intrinsics matrix
     ImgCoords2d

 K - camera intrinsics matrix

perception equation in world frame
----------------------------------
  takes points directly from WorldCoords3D to PxCoords2d
              [u]  
ZP_uv =  Z *  [v]  = K(R P_w + T) = KTP
              [1]
"""


CameraPoseSE3 = Array['4,4', np.float64]
TransformSE3 = Array['4,4', np.float64]
CameraRotationSO3 = Array['3,3', np.float64]
CameraTranslationVector = Array['3', np.float64]
Vector3dHomogenous = Array['4', np.float64]   # TODO: nomenclature difference between having 0 on last coordinate or not

ArrayOfColors = Array['N,3', np.uint8]

WorldCoords3D = Array['N,3', np.float64]      # x away from us, y to right, z up, starts at world origin
CamFlippedWorldCoords3D = Array['N,3', np.float64]   # x right, y down, z out, starts at world origin
CamCoords3d = Array['N,3', np.float64]        # x goes right, y down, z out, looking out of cam, origin is optical center
CamCoords3dHomog = Array['N,3', np.float64]   # X/Z Y/Z 1
ImgCoords2d = Array['N,2', np.float64]        # x goes right, y down, homogenous without z,
PxCoords2d = Array['N,2', np.int64]           # int, x goes down, y goes right, divided by fx/fy, subtracted cx, cy
Cv2PxCoords2d = Array['N,2', np.int64]        # int, x goes right (todo left ?), y goes down

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

ReprojectionErrorVector = Array['2', np.float64]
