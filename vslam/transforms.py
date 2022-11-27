import numpy as np

from utils.custom_types import Array


"""
* perception equation in the camera frame
     [u]    [ f_x,   0, c_x]    [x 
Z *  [v] =  [   0, f_y, c_y]  @ [y = def = K @ P
     [1]    [   0,   0,   1]    [z
     ^       ^
     :        Camera Intrinsics matrix
     ImgCoords2d
     
 K - camera intrinsics matrix
 
* perception equation in world frame
              [u]  
ZP_uv =  Z *  [v]  = K(R P_w + T) = KTP
              [1]
 ZP 
 
 
 * normalized camera coordinates (
 
"""

PxCoords2d = ...    # int, x goes down, y goes right
Cv2PxCoords2d = ...    # int, x goes right, y goes down
# PxCoords3dHomog = ...    # shouldn't exist, too weird
ImgCoords2d = ...       # float, homogenous without z,
ImgCoords3d = ...       # float, z*u, z*v, z
CamCoords3d = ...        # float, X Y Z
CamCoords3dHomog = ...   # float, X/Z Y/Z 1

CameraIntrinsicMatrix = Array['3,3', np.float64]
CameraPoseSE3 = Array['4,4', np.float64]
WorldPoint3D = ...

"""
WorldPoint3D
    -[*CameraPoseSE3]->
        CamCoords3d
            -[/Z] -> 
            CamCoords3dHomog 
                -[* CameraIntrinsicMatrix]-> 
                    ImgCoords3d
                        -[drop third dim]->
                            ImgCoords2d
                                -[-center, *px scale]
                                    -> PxCoords2d
"""
