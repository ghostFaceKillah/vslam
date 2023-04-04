import os

import cv2
import numpy as np
import pandas as pd

from defs import ROOT_DIR
from plotting import Padding, Row, Col, TextRenderer
from utils.custom_types import BGRColor
from utils.cv2_but_its_typed import cv2_circle
from utils.image import take_crop_around, magnify
from vslam.datasets.simdata import SimDataStreamer
from vslam.features import OrbBasedFeatureMatcher
from vslam.poses import get_SE3_pose
from vslam.transforms import px_2d_to_cam_coords_3d_homo
from vslam.triangulation import naive_triangulation


def depth_to_color(depth: float, min_depth: float, max_depth: float) -> BGRColor:
    unit_depth = (np.clip(depth, min_depth, max_depth) - min_depth) / (max_depth - min_depth)
    b = int(unit_depth * 255.)
    g = 0
    r = int((1 - unit_depth) * 255.)
    return b, g, r


if __name__ == '__main__':

    dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-02-04--17-08-25.msgpack')
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    cam_intrinsics = data_streamer.get_cam_intrinsics()
    matcher = OrbBasedFeatureMatcher.build()

    distance_between_eyes = data_streamer.recorded_data.camera_specs.distance_between_eyes

    T2 = get_SE3_pose(x=-distance_between_eyes)

    layout = Col(
        Row(Padding("desc")),
        Row(Padding('left_crop'), Padding('left'), Padding('right'), Padding('right_crop')),
    )

    for obs in data_streamer.stream():
        im_left = obs.left_eye_img
        im_right = obs.right_eye_img

        feature_matches = matcher.detect_and_match_binocular(im_left, im_right)

        # feature matches are in PxCoords. We need to bring them to CamCoords3dHomog

        from_kp_px_coords_2d = np.array([fm.get_from_keypoint_px() for fm in feature_matches], dtype=np.int64)
        from_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(from_kp_px_coords_2d, cam_intrinsics)

        to_kp_px_coords_2d = np.array([fm.get_to_keypoint_px() for fm in feature_matches], dtype=np.int64)
        to_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(to_kp_px_coords_2d, cam_intrinsics)

        depths = naive_triangulation(
            points_in_cam_one=from_kp_cam_coords_3d_homo,
            points_in_cam_two=to_kp_cam_coords_3d_homo,
            cam_one_in_two=T2
        )

        depths_df = pd.Series(depths)

        # draw the matches
        from_canvas_img = np.copy(im_left)
        to_canvas_img = np.copy(im_right)

        dot_thick = 2
        dot_radius = 3

        for i in range(len(feature_matches)):
            match = feature_matches[i]
            depth = depths[i]
            if depth is not None:
                color = depth_to_color(depth, min_depth=3.0, max_depth=25.0)
                cv2_circle(from_canvas_img, match.get_from_keypoint_px()[::-1], color=color, radius=dot_radius, thickness=dot_thick)

        for i, match in enumerate(feature_matches):
            from_img = np.copy(from_canvas_img)
            to_img = np.copy(to_canvas_img)

            crop_from = take_crop_around(im_left, around_point=match.get_from_keypoint_px(), crop_size=(32, 32))
            crop_to = take_crop_around(im_right, around_point=match.get_to_keypoint_px(), crop_size=(32, 32))

            depth = depths[i]
            if depth is not None:
                color = depth_to_color(depth, min_depth=3.0, max_depth=25.0)

            cv2_circle(from_img, match.get_from_keypoint_px()[::-1], color=color, radius=10, thickness=4)
            cv2_circle(to_img, match.get_to_keypoint_px()[::-1], color=color, radius=10, thickness=4)

            depth_txt = f" Est depth = {depth:.2f}" if depth is not None else f" Est depth = None"

            desc = f"Match {i} out of {len(feature_matches)}. Euc dist = {match.get_pixel_distance():.2f} " \
                   f"Hamming dist = {match.get_hamming_distance():.2f}" + depth_txt

            img = layout.render({
                'desc': TextRenderer().render(desc),
                'left': from_img,
                'right': to_img,
                'left_crop': magnify(crop_from, factor=4.0),
                'right_crop': magnify(crop_to, factor=4.0),
            })

            # cv2.imwrite(f'imgs/feature_matching_{i:04d}.png', magnify(img, factor=0.7))
            cv2.imshow('wow', img)
            cv2.waitKey(-1)



