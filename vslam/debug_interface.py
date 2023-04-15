from typing import List, Protocol

from vslam.features import FeatureMatch
from vslam.ketyframe import Keyframe


class IProvidesKeyframe(Protocol):
    def get_keyframe(self) -> Keyframe:
        ...


class IProvidesFeatureMatches(Protocol):
    def get_feature_matches(self) -> List[FeatureMatch]:
        ...

