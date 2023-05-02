import os

import cv2

from defs import ROOT_DIR
from vslam.datasets.simdata import SimDataStreamer
from vslam.debug import FeatureMatchDebugger
from vslam.features import OrbBasedFeatureMatcher

if __name__ == "__main__":
    dataset_path = os.path.join(
        ROOT_DIR, "data/short_recording_2023-04-20--22-46-06.msgpack"
    )

    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    debugger = FeatureMatchDebugger.from_defaults()

    for obs in data_streamer.stream():
        matcher = OrbBasedFeatureMatcher.build()

        im_left = obs.left_eye_img
        im_right = obs.right_eye_img

        feature_matches = matcher.detect_and_match_binocular(im_left, im_right)

        print(f"Found {len(feature_matches)} feature matches")

        for img in debugger.render(im_left, im_right, feature_matches):
            # cv2.imwrite(f'imgs/feature_matching_{i:04d}.png', magnify(img, factor=0.7))
            cv2.imshow("wow", img)
            cv2.waitKey(-1)
