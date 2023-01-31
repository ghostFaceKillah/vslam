import cv2
import numpy as onp
from jax import numpy as np

from sim.birds_eye_view_render import get_view_spcifier_from_scene, render_birdseye_view
from sim.clipping import ClippingSurfaces
from sim.egocentric_render import render_scene_pixelwise_depth
from sim.sample_scenes import get_triangles_in_sky_scene_2
from sim.ui import key_to_maybe_transforms
from utils.colors import BGRCuteColors
from utils.profiling import just_time
from vslam.math import normalize_vector
from vslam.poses import get_SE3_pose
from vslam.types import CameraIntrinsics, CameraPoseSE3

if __name__ == '__main__':
    screen_h = 480
    screen_w = 640

    sky_color = BGRCuteColors.SKY_BLUE
    ground_color = tuple(x - 20 for x in BGRCuteColors.CYAN)

    # higher f_mod -> less distortion, less field of view
    f_mod = 2.0

    shade_color = BGRCuteColors.DARK_GRAY
    cam_intrinsics = CameraIntrinsics(
        fx=screen_w / 4 * f_mod,
        fy=screen_h / 3 * f_mod,
        cx=screen_w / 2,
        cy=screen_h / 2,
    )
    light_direction = normalize_vector(np.array([1.0, -1.0, -8.0]))
    clipping_surfaces = ClippingSurfaces.from_screen_dimensions_and_cam_intrinsics(screen_h, screen_w, cam_intrinsics)

    # looking toward +x direction in world frame, +z in camera
    camera_pose: CameraPoseSE3 = get_SE3_pose(x=-2.5)

    # triangles = get_two_triangle_scene()
    # triangles = get_cube_scene()
    # triangles = get_triangles_in_sky_scene()
    triangles = get_triangles_in_sky_scene_2()
    view_specifier = get_view_spcifier_from_scene(triangles)

    i = 0
    while True:
        with just_time('render'):
            screen = render_scene_pixelwise_depth(
                screen_h, screen_w,
                camera_pose,
                triangles,
                cam_intrinsics,
                light_direction,
                sky_color,
                ground_color,
                shade_color,
                clipping_surfaces
            )

        bev = render_birdseye_view(
            screen_h=screen_h,
            screen_w=screen_w,
            view_specifier=view_specifier,
            camera_pose=camera_pose,
            camera_intrinsics=cam_intrinsics,
            triangles=triangles,
            bg_color=ground_color
        )

        cv2.imshow('birdseye', onp.array(bev))
        cv2.imshow('scene', onp.array(screen))
        key = cv2.waitKey(-1)

        if key == 27:
            print("caught escape key, exiting")
            break

        # mutate state based on keys
        transforms = key_to_maybe_transforms(key)

        if transforms.scene is not None:
            triangles = [triangle.mutate(transforms.scene) for triangle in triangles]

        if transforms.camera is not None:
            camera_pose = camera_pose @ transforms.camera

        i += 1
