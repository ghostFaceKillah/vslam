from typing import List, Tuple

import attr
import cv2
import numpy as np
import pandas as pd

from utils.custom_types import BinaryFeature, BGRImageArray, Array
from utils.profiling import just_time


@attr.s(auto_attribs=True)
class FeatureMatch:
    raw_match: cv2.DMatch    # trainIdx, queryIdx, distance # I am assuming hamming distance
    from_keypoint: cv2.KeyPoint   # pt, size, angle, octave, class_id, response
    to_keypoint: cv2.KeyPoint
    from_feature: BinaryFeature  # binary feature
    to_feature: BinaryFeature   # binary feature

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
        return self.raw_match.distance

    def get_pixel_distance(self) -> float:
        from_pt = self.from_keypoint.pt
        to_pt = self.to_keypoint.pt
        return np.sqrt((from_pt[0] - to_pt[0])**2 + (from_pt[1] - to_pt[1])**2)

    def get_from_keypoint_px(self) -> Tuple[int, int]:
        fw, fh = self.from_keypoint.pt    # mind the opencv coord flip
        # mildly buggy :) something something rounding
        return int(fh), int(fw)

    def get_to_keypoint_px(self) -> Tuple[int, int]:
        fw, fh = self.to_keypoint.pt    # mind the opencv coord flippendo
        # mildly buggy :) something something rounding
        return int(fh), int(fw)


def analyze_orb_feature_matches(matches: List[FeatureMatch]):
    df = pd.DataFrame({
        'px_dist': [match.get_pixel_distance() for match in matches],
        'hamming_dist': [match.get_hamming_distance() for match in matches]
    })
    print(df.quantile(np.linspace(0, 1, num=21)))


@attr.define
class OrbFeatureDetections:
    descriptors: Array['N,32', np.uint8]
    keypoints: List[cv2.KeyPoint]


@attr.s(auto_attribs=True)
class OrbBasedFeatureMatcher:
    orb_feature_detector: cv2.ORB
    feature_matcher: cv2.BFMatcher
    max_px_distance: float = 100.0
    max_hamming_distance: float = 3

    @classmethod
    def build(
        cls,
        max_features: int = 10000,
        max_px_distance: float =100.0,
        max_hamming_distance: float = 31    # this will likely break during turning. In the book they had
    ):
        orb_feature_detector = cv2.ORB_create(max_features)
        feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        return cls(
            orb_feature_detector=orb_feature_detector,
            feature_matcher=feature_matcher,
            max_px_distance=max_px_distance,
            max_hamming_distance=max_hamming_distance
        )

    def detect(self, img: BGRImageArray) -> OrbFeatureDetections:
        keypoints, descriptors = self.orb_feature_detector.detectAndCompute(img, None)
        return OrbFeatureDetections(
            descriptors=descriptors,
            keypoints=keypoints
        )

    def match(self, left_detections: OrbFeatureDetections, right_detections: OrbFeatureDetections)-> List[FeatureMatch]:
        with just_time('matching'):
            raw_matches = self.feature_matcher.match(left_detections.descriptors, np.array(right_detections.descriptors))

        matches = [
            FeatureMatch.from_cv2_match_and_keypoints(
                match=match,
                from_keypoints=left_detections.keypoints,
                to_keypoints=right_detections.keypoints,
                from_features=left_detections.descriptors,
                to_features=right_detections.descriptors
            )
            for match in raw_matches
        ]

        filtered_matches = [
            match for match in matches
            if match.get_pixel_distance() < self.max_px_distance and
               match.get_hamming_distance() <= self.max_hamming_distance
        ]

        sorted_matches = sorted(filtered_matches, key=lambda match: match.get_hamming_distance())
        return sorted_matches

    def detect_and_match_binocular(
        self,
        img_left: BGRImageArray,
        img_right: BGRImageArray,
    ) -> List[FeatureMatch]:

        with just_time('detecting'):
            left_detections = self.detect(img_left)
            right_detections = self.detect(img_right)

        return self.match(left_detections, right_detections)


