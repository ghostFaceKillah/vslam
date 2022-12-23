"""
Small demo of naive rendering
"""
import attr
import cv2
import numpy as np

from utils.colors import BGRCuteColors
from utils.custom_types import Array
from utils.image import get_canvas
from vslam.math import vector_to_unit, dot_product
from vslam.poses import get_SE3_pose
from vslam.transforms import get_world_to_cam_coord_flip_matrix, SE3_inverse, the_cv_flip
from vslam.types import CameraPoseSE3, CameraIntrinsics, Vector3dHomogenous


# https://github.com/krishauser/Klampt/blob/master/Python/klampt/math/se3.py
# cool reference

@attr.define
class Triangle3d:
    """ a triangle floating in space """
    points: Array['3,4', np.float64]   # three 3d points

    def get_surface_normal(self) -> Vector3dHomogenous:
        # TODO: There has to be a convention around those in rendering community
        vec_1 = (self.points[1] - self.points[0])[:3]
        vec_2 = (self.points[2] - self.points[0])[:3]
        resu = np.cross(vec_1, vec_2)
        return resu / np.linalg.norm(resu)


if __name__ == '__main__':
    screen_h = 640
    screen_w = 480

    camera_pose: CameraPoseSE3 = get_SE3_pose()
    # looking toward -y direction in world frame

    world_to_cam_flip = get_world_to_cam_coord_flip_matrix()

    # we have a triangle in space
    triangle = Triangle3d(np.array([
        [1.5, -1.0, -1.0, 1.0],
        [1.5,  2.0, -1.0, 1.0],
        [1.5, -1.0, 2.0, 1.0],
    ], dtype=np.float64))

    # we bring points to cam coordinate frame
    # 1) camera pose transform
    # 2) coordinate flip
    # cam_coordinates * points_in_cam = world_coordinates * points_in_world
    # points_in_cam = world_to_cam_matrix() * inverse_of_cam_pose * points_in_world
    cam_pose_inv = SE3_inverse(camera_pose)
    cam_points_T = world_to_cam_flip @ cam_pose_inv @ triangle.points.T
    cam_points = cam_points_T.T

    # drop_homogenous coord one
    cam_points = cam_points[:, :3]

    # drop things that are not visible
    cam_points = cam_points[cam_points[:, 2] > 1]

    # project to unit image plane
    projected_to_unit_plane = cam_points / cam_points[:, 2][:, np.newaxis]

    # map to pixel coordinates

    # I want image to be 640 by 480
    # I want it to map from 4 by 3 meters
    # TODO: make a factory function
    cam_intrinsics = CameraIntrinsics(fx=640 / 4, fy=480 / 3, cx=640 / 2, cy=480 / 2)

    px_coords = (cam_intrinsics.get_homo_cam_coords_to_px_coords_matrix()  @ projected_to_unit_plane.T).T

    # render the triangle if at least one point lies within the image plane
    ...

    # we will want to compute surface normal by computing crossproduct
    normal = triangle.get_surface_normal()
    # we will want to compute color by inner_product(light direction, surface normal)
    light_direction = vector_to_unit(np.array([1.0, -1.0, -8.0]))
    dot_product(normal, light_direction)
    color = BGRCuteColors.GRASS_GREEN

    # we will want to render triangle on surface plane

    # prepare surface plane
    screen = get_canvas(shape=(480, 640, 3), background_color=BGRCuteColors.DARK_GRAY)

    # draw triangle to screen
    """
    .   @param img Image.
    .   @param pts Array of polygons where each polygon is represented as an array of points. List[Tuple[int, int]]
    .   @param color Polygon color.
    .   @param lineType Type of the polygon boundaries. See #LineTypes
    .   @param shift Number of fractional bits in the vertex coordinates.
    .   @param offset Optional offset of all points of the contours.
    """

    poly_coords = the_cv_flip(px_coords.round().astype(np.int32))

    cv2.fillPoly(screen, [poly_coords], color)
    cv2.imshow('scene', screen)
    cv2.waitKey(-1)

    """
    to jest mój framework do renderowania :) 
    opengl from scratch 
    """



