import attr
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



from typing import List

from custom_types import DirPath, FilePath, Array, BinaryFeature
from utils import batched


def get_im_path(
    dataset_path: DirPath = '/Users/misiu-dev/temp/phd/kitti-dataset/sequences',
    sequence_no: int = 0,
    cam_no: int = 0,
    image_no: int = 0
) -> FilePath:
    assert 0 <= sequence_no <= 21
    assert 0 <= cam_no <= 1

    return os.path.join(dataset_path, f"{sequence_no:02d}", f"image_{cam_no:d}", f"{image_no:06d}.png")


def get_calibration_path(
        dataset_path: DirPath = '/Users/misiu-dev/temp/phd/kitti-dataset/sequences',
        sequence_no: int = 0,
) -> FilePath:
    assert 0 <= sequence_no <= 21
    return os.path.join(dataset_path, f"{sequence_no:02d}", "calib.txt")


@attr.s(auto_attribs=True)
class Calibration:
    camera_left_projection_matrix: np.ndarray['3,4', np.float64]
    camera_right_projection_matrix: np.ndarray['3,4', np.float64]


def read_calib_from_file(filepath: FilePath) -> Calibration:
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    P_rect_00 = np.reshape(data['P0'], (3, 4))  # these are the two relevant ones
    P_rect_10 = np.reshape(data['P1'], (3, 4))  # these are the two relevant ones
    # these are two identical cameras
    # 10 is offset by 386.1448 mm toward the driver's side (it's negative in the x dimension)

    # I guess I should just place them directly in the constants
    # P_rect_20 = np.reshape(data['P2'], (3, 4))
    # P_rect_30 = np.reshape(data['P3'], (3, 4))

    # https://github.com/pratikac/kitti/blob/master/readme.raw.txt
    #  P_rect_xx: 3x4 projection matrix after rectification

    return Calibration(
        camera_right_projection_matrix=P_rect_00,
        camera_left_projection_matrix=P_rect_10
    )


@attr.s(auto_attribs=True)
class FeatureMatch:
    _raw_match: cv2.DMatch    # trainIdx, queryIdx, distance # I am assuming hamming distance
    _from_keypoint: cv2.KeyPoint   # pt, size, angle, octave, class_id, response
    _to_keypoint: cv2.KeyPoint
    _from_feature: Array['N', np.uint8]   # binary feature
    _to_feature: Array['N', np.uint8]    # binary feature

    @classmethod
    def from_cv2_match_and_keypoints(
        cls,
        match: cv2.DMatch,
        from_keypoints: List[cv2.KeyPoint],
        to_keypoints: List[cv2.KeyPoint],
        from_features: List[BinaryFeature],
        to_features: List[BinaryFeature],
    ) -> 'FeatureMatch':
        return cls(
            raw_match=match,
            from_keypoint=from_keypoints[match.queryIdx],
            to_keypoint=to_keypoints[match.trainIdx],
            from_feature=from_features[match.queryIdx],
            to_feature=to_features[match.trainIdx]
        )

    def get_hamming_distance(self) -> float:
        return self._raw_match.distance

    def get_pixel_distance(self) -> float:
        from_pt = self._from_keypoint.pt
        to_pt = self._to_keypoint.pt
        return np.sqrt((from_pt[0] - to_pt[0])**2 + (from_pt[1] - to_pt[1])**2)



def draw_matches(
        im_left,
        im_right,
):
    pass




if __name__ == '__main__':
    '/Users/misiu-dev/temp/phd/kitti-dataset/'
    '/Users/misiu-dev/temp/phd/kitti-dataset/sequences'

    # read the data
    # like 2 iamges
    # maybe triangulate it or sth
    calibration = read_calib_from_file(get_calibration_path(sequence_no=0))

    sequence_no = 0
    im_left = cv2.imread(get_im_path(sequence_no=sequence_no, cam_no=1))
    im_right = cv2.imread(get_im_path(sequence_no=sequence_no, cam_no=0))

    orb = cv2.ORB_create(100000)

    img_left_kp, img_left_desc = orb.detectAndCompute(im_left, None)
    img_right_kp, img_right_desc = orb.detectAndCompute(im_right, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.match(img_left_desc, img_right_desc)

    matches = [
        FeatureMatch.from_cv2_match_and_keypoints(
            match=match,
            from_keypoints=img_left_kp,
            to_keypoints=img_right_kp,
            from_features=img_left_desc,
            to_features=img_right_desc
        )
        for match in raw_matches
    ]

    # df = pd.Series([match.distance for match in raw_matches])
    df = pd.Series([match.get_pixel_distance() for match in matches])
    print(df.quantile(np.linspace(0, 1, num=21)))
    # let's take maybe ~30pixels max distance. When turning, that will probably be different
    max_px_distance = 30.0

    relevant_raw_matches = []
    for match, raw_match in zip(matches, raw_matches):
        if match.get_pixel_distance() < max_px_distance:
            relevant_raw_matches.append(raw_match)

    print(f"Pre filtering = {len(raw_matches)} post filtering = {len(relevant_raw_matches)}")

    for sub_matches in batched(relevant_raw_matches, 1):
        final_img = cv2.drawMatches(
            im_left,
            img_left_kp,
            im_right,
            img_right_kp,
            sub_matches,
            2
        )
        cv2.imshow("Matches", final_img)
        cv2.waitKey(-1)


    # Idea: really nice visualization for the matches

    x = 1

