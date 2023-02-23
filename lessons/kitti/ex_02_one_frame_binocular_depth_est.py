import cv2
import numpy as np
import pandas as pd

from plotting import Col, Padding
from utils.colors import BGRCuteColors
from utils.custom_types import BGRColor
from utils.cv2_but_its_typed import cv2_circle
from vslam.datasets.kitti import KittiDataset
from vslam.features import OrbBasedFeatureMatcher
from vslam.transforms import px_2d_to_cam_coords_3d_homo
from vslam.triangulation import naive_triangulation


def depth_to_color(depth: float, min_depth: float, max_depth: float) -> BGRColor:
    unit_depth = (np.clip(depth, min_depth, max_depth) - min_depth) / (max_depth - min_depth)
    b = int(unit_depth * 255.)
    g = 0
    r = int((1 - unit_depth) * 255.)
    return b, g, r


if __name__ == '__main__':

    dataset = KittiDataset(sequence_no=0)
    im_left = dataset.get_left_image(image_no=0)
    im_right = dataset.get_right_image(image_no=0)
    cam_intrinsics = dataset.get_left_camera_intrinsics()
    # calib = dataset.get_calibration()

    matcher = OrbBasedFeatureMatcher.build()
    # TODO: Ugly flip, gotta get back to this - we need this because of Kitti calibration format (right cam is the 0, 0)
    feature_matches = matcher.detect_and_match_binocular(im_left, im_right)

    # feature matches are in PxCoords. We need to bring them to CamCoords3dHomog

    from_kp_px_coords_2d = np.array([fm.get_from_keypoint_px() for fm in feature_matches], dtype=np.int64)
    from_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(from_kp_px_coords_2d, cam_intrinsics)

    to_kp_px_coords_2d = np.array([fm.get_to_keypoint_px() for fm in feature_matches], dtype=np.int64)
    to_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(to_kp_px_coords_2d, cam_intrinsics)

    # TODO: too messy kitti initializaiton, gotta go back to it
    # specifically kitti transormation matrix has pixel scaling in it
    T2 = np.array([
        [1.0, 0.0, 0.0, -0.3861448],
        [0.0, 1.0, 0.0, 0.],
        [0.0, 0.0, 1.0, 0.],
        [0.0, 0.0, 0.0, 1.],
    ])

    depths = naive_triangulation(
        points_in_cam_one=from_kp_cam_coords_3d_homo,
        points_in_cam_two=to_kp_cam_coords_3d_homo,
        cam_two_in_cam_one=T2
    )

    depths_df = pd.Series(depths)

    canvas_img = np.copy(im_left)
    deeper_canvas_img = np.copy(im_left)

    for i in range(len(feature_matches)):
        match = feature_matches[i]
        depth = depths[i]
        if depth is not None:
            color = depth_to_color(depth, min_depth=3.0, max_depth=25.0)
            cv2_circle(canvas_img, match.get_from_keypoint_px()[::-1], color=color, radius=1, thickness=1)

            color = depth_to_color(depth, min_depth=10.0, max_depth=35.0)
            cv2_circle(deeper_canvas_img, match.get_from_keypoint_px()[::-1], color=color, radius=1, thickness=1)
        else:
            cv2_circle(canvas_img, match.get_from_keypoint_px()[::-1], color=BGRCuteColors.DARK_GRAY, radius=1, thickness=1)
            cv2_circle(deeper_canvas_img, match.get_from_keypoint_px()[::-1], color=BGRCuteColors.DARK_GRAY, radius=1, thickness=1)

    layout = Col(Padding('deep'), Padding('deeper'))

    img = layout.render({
        'deep': canvas_img,
        'deeper': deeper_canvas_img,
    })

    cv2.imshow('wow', img)
    cv2.waitKey(-1)

