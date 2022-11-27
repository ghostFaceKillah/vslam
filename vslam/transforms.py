import numpy as np
from utils.custom_types import Array

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

CameraIntrinsicMatrix = Array['3,3', np.float64]
CameraPoseSE3 = Array['4,4', np.float64]
CameraRotationSO3 = Array['3,3', np.float64]

WorldCoords3D = Array['N,3', np.float64]       # starts at world origin
CamCoords3d = Array['N,3', np.float64]        # x goes left, y goes down, z goes towards viewer
CamCoords3dHomog = Array['N,3', np.float64]   # X/Z Y/Z 1
UndistCamCoords3dHomog = Array['N,3', np.float64]  # same as CamCoords3dHomog, but not linear for once
ImgCoords3d = Array['N,3', np.float64]        # x goes left, y down, z out of cam; z*u, z*v, z
ImgCoords2d = Array['N,2', np.float64]        # x goes left, y down, z out of cam; homogenous without z,
PxCoords2d = Array['N,2', np.int64]           # int, x goes down, y goes right
Cv2PxCoords2d = Array['N,2', np.int64]        # int, x goes right, y goes down

Vector3d = Array['3', np.float64]


"""
WorldCoords3D
    -[CameraPoseSE3 * _]->
        CamCoords3d
            -[/Z] -> 
                CamCoords3dHomog 
                    -[undistortion, perhaps]->
                        UndistCamCoords3dHomog 
                            -[* CameraIntrinsicMatrix]-> 
                                ImgCoords3d
                                    -[drop third dim]->
                                        ImgCoords2d
                                            -[-center, *px scale]
                                                -> PxCoords2d
"""
