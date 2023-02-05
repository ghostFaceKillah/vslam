import os

import cv2
import numpy as np
import pandas as pd

from defs import ROOT_DIR
from plotting import Padding, Row
from utils.custom_types import BGRColor
from utils.cv2_but_its_typed import cv2_circle
from vslam.datasets.simdata import SimDataStreamer
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

    dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-02-04--17-08-25.msgpack')

    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)
    cam_intrinsics = data_streamer.get_cam_intrinsics()
    matcher = OrbBasedFeatureMatcher.build()

    # data_streamer.recorded_data.camera_specs.distance_between_eyes

    distance_between_eyes = data_streamer.recorded_data.camera_specs.distance_between_eyes

    T2 = np.array([
        [1.0, 0.0, 0.0, distance_between_eyes],
        [0.0, 1.0, 0.0, 0.],
        [0.0, 0.0, 1.0, 0.],
        [0.0, 0.0, 0.0, 1.],
    ])

    for obs in data_streamer.stream():
        im_left = obs.left_eye_img
        im_right = obs.right_eye_img

        feature_matches = matcher.detect_and_match(im_left, im_right)

        # feature matches are in PxCoords. We need to bring them to CamCoords3dHomog

        from_kp_px_coords_2d = np.array([fm.get_from_keypoint_px() for fm in feature_matches], dtype=np.int64)
        from_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(from_kp_px_coords_2d, cam_intrinsics)

        to_kp_px_coords_2d = np.array([fm.get_to_keypoint_px() for fm in feature_matches], dtype=np.int64)
        to_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(to_kp_px_coords_2d, cam_intrinsics)

        depths = naive_triangulation(
            pts_1=from_kp_cam_coords_3d_homo,
            pts_2=to_kp_cam_coords_3d_homo,
            T2=T2
        )

        depths_df = pd.Series(depths)

        canvas_img = np.copy(im_left)
        deeper_canvas_img = np.copy(im_left)

        dot_thick = 2
        dot_radius = 2

        for i in range(len(feature_matches)):
            match = feature_matches[i]
            depth = depths[i]
            if depth is not None:
                color = depth_to_color(depth, min_depth=3.0, max_depth=25.0)
                cv2_circle(canvas_img, match.get_from_keypoint_px()[::-1], color=color, radius=dot_radius, thickness=dot_thick)

        layout = Row(Padding('depth'))

        img = layout.render({
            'depth': canvas_img,
        })

        cv2.imshow('wow', img)
        cv2.waitKey(-1)

