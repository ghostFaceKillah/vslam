from typing import List, Iterable, Optional

import attr
import cv2
import numpy as np
import pandas as pd

from plotting import Packer, Col, Row, Padding, TextRenderer
from utils.colors import BGRCuteColors
from utils.custom_types import BinaryFeature, BGRImageArray, Array, Pixel
from utils.cv2_but_its_typed import cv2_circle
from utils.image import take_crop_around, magnify
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

    def get_from_keypoint_px(self) -> Pixel:
        fw, fh = self.from_keypoint.pt    # mind the opencv coord flip
        # mildly buggy :) something something rounding
        return int(fh), int(fw)

    def get_to_keypoint_px(self) -> Pixel:
        fw, fh = self.to_keypoint.pt    # watch out: we go from OpenCVPixel to Pixel
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
    max_px_distance: float
    max_hamming_distance: float

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

    def _describe_match_quality_distribution(self, matches: List[FeatureMatch]):
        # don't do this at home
        import pandas as pd

        px_dists = []
        hamming_dists = []

        for match in matches:
            px_dists.append(match.get_pixel_distance())
            hamming_dists.append(match.get_hamming_distance())

        df = pd.DataFrame({'px_dists': px_dists, 'hamming_dists': hamming_dists})
        print(df.describe())

    def match(
            self,
            left_detections: OrbFeatureDetections,
            right_detections: OrbFeatureDetections,
            debug_matches: bool = False
        ) -> List[FeatureMatch]:
        with just_time('matching'):
            raw_cv_matches = self.feature_matcher.match(left_detections.descriptors, np.array(right_detections.descriptors))

        matches = [
            FeatureMatch.from_cv2_match_and_keypoints(
                match=match,
                from_keypoints=left_detections.keypoints,
                to_keypoints=right_detections.keypoints,
                from_features=left_detections.descriptors,
                to_features=right_detections.descriptors
            )
            for match in raw_cv_matches
        ]

        if debug_matches:
            self._describe_match_quality_distribution(matches)

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


@attr.define
class FeatureMatchDebugger:
    ui_layout: Packer

    @classmethod
    def from_defaults(cls):
        layout = Col(
            Row(Padding("desc")),
            Row(Padding('left_crop'), Padding('left'), Padding('right'), Padding('right_crop')),
        )

        return cls(ui_layout=layout)

    def render(
        self,
        from_img: BGRImageArray,
        to_img: BGRImageArray,
        matches: List[FeatureMatch],
        depths: Optional[List[float]] = None
    ) -> Iterable[BGRImageArray]:
        # draw the matches
        from_canvas_img = np.copy(from_img)
        to_canvas_img = np.copy(to_img)

        assert depths is None or len(depths) == len(matches), f'{len(depths)=} != {len(matches)=}'

        if depths is None:
            depth_txts = [''] * len(matches)
        else:
            depth_txts = ['diverged' if depth is None else f'depth: {depth:.2f}' for depth in depths]

        for match in matches:
            cv2_circle(from_canvas_img, match.get_from_keypoint_px()[::-1], color=BGRCuteColors.GRASS_GREEN, radius=1,
                       thickness=1)
            cv2_circle(to_canvas_img, match.get_to_keypoint_px()[::-1], color=BGRCuteColors.GRASS_GREEN, radius=1,
                       thickness=1)

        for i, (match, depth_txt) in enumerate(zip(matches, depth_txts)):
            from_img_ = np.copy(from_canvas_img)
            to_img_ = np.copy(to_canvas_img)

            crop_from = take_crop_around(from_canvas_img, around_point=match.get_from_keypoint_px(), crop_size=(32, 32))
            crop_to = take_crop_around(to_canvas_img, around_point=match.get_to_keypoint_px(), crop_size=(32, 32))

            cv2_circle(from_img_, match.get_from_keypoint_px()[::-1], color=BGRCuteColors.ORANGE, radius=10, thickness=4)
            cv2_circle(to_img_, match.get_to_keypoint_px()[::-1], color=BGRCuteColors.ORANGE, radius=10, thickness=4)

            desc = f"Match {i} out of {len(matches)}. Euc dist = {match.get_pixel_distance():.2f} " \
                   f"Hamming dist = {match.get_hamming_distance():.2f} " + depth_txt

            img = self.ui_layout.render({
                'desc': TextRenderer().render(desc),
                'left': from_img_,
                'right': to_img_,
                'left_crop': magnify(crop_from, factor=4.0),
                'right_crop': magnify(crop_to, factor=4.0),
            })

            yield img