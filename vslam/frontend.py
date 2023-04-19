""" Frontend means that we are doing 'local visual odometry'
We visually match features to keyframe and we try to figure out how much we have moved with respect to that.
We do not do any global state correction - no loop closure and no global state optimization.
"""
from typing import List, Optional, Tuple

import attr
import numpy as np

from sim.sim_types import RenderTriangle3d, CameraSpecs, Observation
from vslam.debug_interface import IProvidesFeatureMatches, IProvidesKeyframe
from vslam.features import OrbBasedFeatureMatcher, FeatureMatch
from vslam.ketyframe import Keyframe, KeyframeMatchPoseTrackingResult, estimate_keyframe, estimate_pose_wrt_keyframe
from vslam.tracking import VelocityPoseTracker
from vslam.types import TransformSE3, CameraPoseSE3


class FrontendState:
    """ In future maybe add LOST or sth"""
    class Init:
        """ Just initialized, we need to establish a keyframe. """
        ...

    @attr.define
    class Tracking:
        """ We are tracking, but maybe some iterations we have suspicions
        that the pose matches are not great quality """
        frames_since_keyframe = attr.ib(type=int)
        keyframe = attr.ib(type=Keyframe, repr=False)


@attr.s(auto_attribs=True)
class FrontendStaticDebugData:
    """ Privileged information for the sake of debug """
    scene: List[RenderTriangle3d]


@attr.s(auto_attribs=True)
class FrontendResultDebugData(IProvidesKeyframe, IProvidesFeatureMatches):
    frames_since_keyframe: int
    keyframe: Keyframe
    feature_matches: List[FeatureMatch]


@attr.s(auto_attribs=True)
class FrontendTrackingResult:
    baselink_pose_estimate: TransformSE3

    debug_data: FrontendResultDebugData


class TrackingQualityEstimate:
    @attr.define
    class Healthy:
        ...

    @attr.define
    class Bad:
        reasons: str


@attr.define
class FrontendPoseQualityEstimator:
    minimum_number_of_matches: int = 8
    max_allowed_error: float = 0.02

    def estimate_tracking_quality(self, tracking_result: KeyframeMatchPoseTrackingResult) -> bool:
        """ Should roll to a new keyframe ? """

        tracking_is_good = True
        reasons = []

        if (no_matches := len(tracking_result.tracking_quality_info.euclidean_errors)) < self.minimum_number_of_matches:
            tracking_is_good = False
            reasons.append(f"{no_matches=}, which is less then required {self.minimum_number_of_matches=}")

        err_75th_percentile = np.percentile(tracking_result.tracking_quality_info.euclidean_errors, 75)
        if err_75th_percentile > self.max_allowed_error:
            tracking_is_good = False
            reasons.append(f"{err_75th_percentile=}, which is more then {self.max_allowed_error=:.4f}")

        if tracking_is_good:
            return TrackingQualityEstimate.Healthy()
        else:
            return TrackingQualityEstimate.Bad(reasons=" and ".join(reasons))


@attr.s(auto_attribs=True)
class Frontend:
    """ At this point, it just groups up stuff related to Frontend """
    matcher: OrbBasedFeatureMatcher
    cam_specs: CameraSpecs
    pose_tracker: VelocityPoseTracker

    tracking_quality_estimator: FrontendPoseQualityEstimator = attr.Factory(FrontendPoseQualityEstimator)

    keyframe: Optional[Keyframe] = None
    state: FrontendState = attr.Factory(FrontendState.Init)
    debug_data: Optional[FrontendStaticDebugData] = None

    @classmethod
    def from_defaults(
        cls,
        cam_specs: CameraSpecs,
        start_pose: Optional[CameraPoseSE3] = None,
        debug_data: Optional[FrontendStaticDebugData] = None
    ):
        return cls(
            matcher=OrbBasedFeatureMatcher.build(),
            cam_specs=cam_specs,
            pose_tracker=VelocityPoseTracker.from_defaults() if start_pose is None else VelocityPoseTracker(start_pose),
            debug_data=debug_data
        )

    def _estimate_new_keyframe(
        self,
        obs: Observation,
        baselink_pose_estimate: TransformSE3
    ) -> Tuple[FrontendTrackingResult, FrontendState]:
        keyframe, keyframe_debug_data = estimate_keyframe(
            obs=obs,
            matcher=self.matcher,
            baselink_pose=baselink_pose_estimate,
            cam_intrinsics=self.cam_specs.intrinsics,
            cam_extrinsics=self.cam_specs.extrinsics,
            debug_scene=self.debug_data.scene
        )

        resu = FrontendTrackingResult(
            baselink_pose_estimate=baselink_pose_estimate,
            debug_data=FrontendResultDebugData(
                frames_since_keyframe=0,
                keyframe=keyframe,
                feature_matches=keyframe_debug_data.feature_matches,
            )
        )
        state = FrontendState.Tracking(
            frames_since_keyframe=0,
            keyframe=keyframe
        )
        return resu, state

    def _track(self, obs: Observation) -> Tuple[FrontendTrackingResult, FrontendState]:
        # TODO: refactor me pls
        match self.state:
            case FrontendState.Init():
                prior_baselink_pose_estimate = self.pose_tracker.get_next_baselink_in_world_pose_estimate()
                return self._estimate_new_keyframe(obs, prior_baselink_pose_estimate)
            case FrontendState.Tracking(frames_since_keyframe, keyframe):
                prior_baselink_pose_estimate = self.pose_tracker.get_next_baselink_in_world_pose_estimate()
                tracking_result, debug_data = estimate_pose_wrt_keyframe(
                    obs=obs,
                    matcher=self.matcher,
                    cam_specs=self.cam_specs,
                    baselink_pose_estimate_in_world=prior_baselink_pose_estimate,   # TODO: oops
                    keyframe=keyframe
                )
                match tracking_result:
                    case KeyframeMatchPoseTrackingResult.Failure(reason=reason):
                        print(f"Tracking failed due to {reason}")
                        return self._estimate_new_keyframe(obs, prior_baselink_pose_estimate)
                    case KeyframeMatchPoseTrackingResult.Success(posterior_baselink_pose_estimate, _):
                        # posterior_baselink_pose_estimate = tracking_result.pose_estimate
                        tracking_quality_estimate = self.tracking_quality_estimator.estimate_tracking_quality(
                            tracking_result
                        )
                        x = 1

                        match tracking_quality_estimate:
                            case TrackingQualityEstimate.Bad(reasons=reasons):
                                # important detail: note that we use _prior_ estimate. We trust it more.
                                print(f"Estimating new keyframe. Tracking quality looks bad due to {reasons=}")
                                return self._estimate_new_keyframe(obs, prior_baselink_pose_estimate)
                            case TrackingQualityEstimate.Healthy():
                                resu = FrontendTrackingResult(
                                    baselink_pose_estimate=posterior_baselink_pose_estimate,
                                    debug_data=FrontendResultDebugData(
                                        frames_since_keyframe=frames_since_keyframe+1,
                                        keyframe=keyframe,
                                        feature_matches=debug_data.feature_matches,
                                    )
                                )
                                state = FrontendState.Tracking(
                                    frames_since_keyframe=frames_since_keyframe+1,
                                    keyframe=keyframe
                                )
                                return resu, state
                            case _:
                                raise ValueError("Unhandled TrackingQualityEstimate", tracking_quality_estimate)
                    case _:
                        raise ValueError("Unhandled KeyframeMatchPoseTrackingResult", tracking_result)

            case _:
                raise ValueError("Unhandled state", self.state)

    def track(self, obs: Observation) -> FrontendTrackingResult:
        result, state = self._track(obs)
        self.state = state
        self.pose_tracker.track(result.baselink_pose_estimate)
        return result
