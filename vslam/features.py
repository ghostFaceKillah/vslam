from typing import List, Protocol

import attr
import cv2
import numpy as np
import pandas as pd

from custom_types import Array, BinaryFeature, BGRImageArray


@attr.s(auto_attribs=True)
class FeatureMatch:
    raw_match: cv2.DMatch    # trainIdx, queryIdx, distance # I am assuming hamming distance
    from_keypoint: cv2.KeyPoint   # pt, size, angle, octave, class_id, response
    to_keypoint: cv2.KeyPoint
    from_feature: Array['N', np.uint8]   # binary feature
    to_feature: Array['N', np.uint8]    # binary feature

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


def analyze_orb_feature_matches(matches: List[FeatureMatch]):
    df = pd.DataFrame({
        'px_dist': [match.get_pixel_distance() for match in matches],
        'hamming_dist': [match.get_hamming_distance() for match in matches]
    })
    print(df.quantile(np.linspace(0, 1, num=21)))


@attr.s(auto_attribs=True)
class OrbBasedFeatureMatcher:
    orb_feature_detector: cv2.ORB
    feature_matcher: cv2.BFMatcher
    max_px_distance: float = 100.0
    max_hamming_distance: float = 3

    @classmethod
    def build(
        cls,
        max_features: int = 1000,
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

    def detect_and_match(
        self,
        img_left: BGRImageArray,
        img_right: BGRImageArray,
    ) -> List[FeatureMatch]:

        img_left_kp, img_left_desc = self.orb_feature_detector.detectAndCompute(img_left, None)
        img_right_kp, img_right_desc = self.orb_feature_detector.detectAndCompute(img_right, None)
        raw_matches = self.feature_matcher.match(img_left_desc, img_right_desc)

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

        filtered_matches = [
            match for match in matches
            if match.get_pixel_distance() < self.max_px_distance and
               match.get_hamming_distance() <= self.max_hamming_distance
        ]

        sorted_matches = sorted(filtered_matches, key=lambda match: match.get_hamming_distance())
        return sorted_matches

