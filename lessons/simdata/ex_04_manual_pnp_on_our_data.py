"""
Pretty much reimplementation by hand of
# int Frontend::EstimateCurrentPose

Let us reimplement by hand, in an explicit way:
1) creation of the first keyframe
2) tracking of couple first frames after that

then we will do the full nice frontend implementation.
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import attr
import cv2
import numpy as np

from defs import ROOT_DIR
from sim.sim_types import Observation, CameraSpecs, CameraExtrinsics, RenderTriangle3d
from utils.colors import BGRCuteColors
from utils.custom_types import BGRImageArray
from vslam.cam import CameraIntrinsics
from vslam.datasets.simdata import SimDataStreamer
from vslam.debug import FeatureMatchDebugger, TriangulationDebugger, LocalizationDebugger
from vslam.features import OrbBasedFeatureMatcher, OrbFeatureDetections
from vslam.pnp import gauss_netwon_pnp
from vslam.poses import get_SE3_pose, SE3_pose_to_xytheta
from vslam.transforms import px_2d_to_cam_coords_3d_homo, SE3_inverse, px_2d_to_img_coords_2d, \
    homogenize, CAM_TO_WORLD_FLIP, dehomogenize
from vslam.triangulation import naive_triangulation
from vslam.types import WorldCoords3D, CameraPoseSE3, TransformSE3


@attr.define
class _Keyframe:
    """ Draft class, will get changed"""
    pose: CameraPoseSE3

    points_3d_est: WorldCoords3D
    feature_detections: OrbFeatureDetections
    image: BGRImageArray


def estimate_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        baselink_pose: CameraPoseSE3,
        cam_intrinsics: CameraIntrinsics,
        cam_extrinsics: CameraExtrinsics,
        debug_feature_matches: bool = False,
        debug_depth_estimation: bool = False,
        debug_scene: Optional[List[RenderTriangle3d]] = None,   # purely for debug vis of depth
) -> _Keyframe:
    left_cam_pose = baselink_pose @ cam_extrinsics.get_pose_of_left_cam_in_baselink()

    left_detections = matcher.detect(obs.left_eye_img)
    right_detections = matcher.detect(obs.right_eye_img)
    feature_matches = matcher.match(left_detections, right_detections)

    # def estimate depth from feature matches
    from_kp_px_coords_2d = np.array([fm.get_from_keypoint_px() for fm in feature_matches], dtype=np.int64)
    from_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(from_kp_px_coords_2d, cam_intrinsics)

    to_kp_px_coords_2d = np.array([fm.get_to_keypoint_px() for fm in feature_matches], dtype=np.int64)
    to_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(to_kp_px_coords_2d, cam_intrinsics)

    depths = naive_triangulation(
        points_in_cam_one=from_kp_cam_coords_3d_homo,
        points_in_cam_two=to_kp_cam_coords_3d_homo,
        cam_one_in_two=cam_extrinsics.get_pose_of_left_cam_in_right_cam()
    )
    # looks waaay to much
    # what is this depth measured in ?

    if debug_feature_matches:
        debugger = FeatureMatchDebugger.from_defaults()

        for img in debugger.render(obs.left_eye_img, obs.right_eye_img, feature_matches, depths):
            cv2.imshow('wow', img)
            cv2.waitKey(-1)

    points_3d_est = []
    feature_descriptors = []
    keypoints = []

    assert len(depths) == len(from_kp_cam_coords_3d_homo) == len(feature_matches)

    for depth, point_homo, feature_match in zip(depths, from_kp_cam_coords_3d_homo, feature_matches):
        if depth is None:
            continue

        # OK need to flip back to world frame here
        points_3d_est.append(dehomogenize(CAM_TO_WORLD_FLIP @ homogenize(point_homo * depth)))
        feature_descriptors.append(feature_match.from_feature)
        keypoints.append(feature_match.from_keypoint)

    # TODO: debug depths
    # let's do it now!

    if debug_depth_estimation:
        depth_est_debugger = TriangulationDebugger.from_defaults()
        assert debug_scene is not None, "If we wanna visualize depth debug, we need scene triangles"

        img_iterator = depth_est_debugger.render(
            obs.left_eye_img,
            obs.right_eye_img,
            feature_matches,
            depths,
            baselink_pose,
            cam_intrinsics,
            cam_extrinsics,
            debug_scene
        )

        for i, img in enumerate(img_iterator):
            if i > 10:
                break
            cv2.imshow('wow', img)
            cv2.waitKey(-1)

    return _Keyframe(
        image=obs.left_eye_img,
        pose=left_cam_pose,
        points_3d_est=points_3d_est,
        feature_detections=OrbFeatureDetections(np.array(feature_descriptors), keypoints)
    )


@attr.define
class PoseTracker:
    # last_pose: CameraPoseSE3 = attr.Factory(get_SE3_pose)

    def track(self, pose: CameraPoseSE3):
        return pose


@attr.define
class KeyframeMatchPoseTrackingResult:
    pose_estimate: CameraPoseSE3
    tracking_quality_info: ...


def estimate_pose_wrt_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        cam_specs: CameraSpecs,
        baselink_pose_estimate_in_world: TransformSE3,
        keyframe: _Keyframe,
        debug_feature_matches: bool = False
) -> KeyframeMatchPoseTrackingResult:
    left_detections = matcher.detect(obs.left_eye_img)
    matches = matcher.match(keyframe.feature_detections, left_detections)

    if debug_feature_matches:
        debugger = FeatureMatchDebugger.from_defaults()

        for img in debugger.render(keyframe.image, obs.left_eye_img, matches):
            cv2.imshow('matches-keypoint-to-new-frame', img)
            cv2.waitKey(-1)

    points_2d = []
    points_3d = []

    # resolve List[FeatureMatch] into 2d points that match a subset of keyframe.points_3d_est
    for match in matches:
        points_3d.append(keyframe.points_3d_est[match.raw_match.queryIdx])
        point_2d = np.array(match.get_to_keypoint_px()).astype(np.float64)
        points_2d.append(point_2d)

    points_3d = np.array(points_3d)

    left_camera_pose_in_world = baselink_pose_estimate_in_world @ cam_specs.extrinsics.get_pose_of_left_cam_in_baselink()
    camera_pose_guess_in_keyframe = SE3_inverse(keyframe.pose) @ left_camera_pose_in_world

    posterior_left_cam_pose_estimate_in_keyframe = gauss_netwon_pnp(
        camera_pose_initial_guess_in_keyframe=camera_pose_guess_in_keyframe,
        points_3d_in_keyframe=homogenize(points_3d),
        points_2d_in_img=px_2d_to_img_coords_2d(np.array(points_2d), cam_specs.intrinsics),
        verbose=False
    )

    # want: world T baselink
    posterior_baselink_pose_estimate_in_world = (
        keyframe.pose   # world T keyframe
        @ posterior_left_cam_pose_estimate_in_keyframe  # # keyframe T cam
        @ SE3_inverse(cam_specs.extrinsics.get_pose_of_left_cam_in_baselink())   # cam T baselink
    )

    return KeyframeMatchPoseTrackingResult(
        posterior_baselink_pose_estimate_in_world,
        tracking_quality_info=None
    )


@attr.define
class VelocityPoseTracker:
    # it predicts the next pose
    current_pose_estimate: CameraPoseSE3

    @classmethod
    def from_defaults(cls):
        return cls(current_pose_estimate=np.eye(4, np.float64))

    def track(self, new_pose: CameraPoseSE3):
        # TODO: propagate it forward by one step
        self.current_pose_estimate = new_pose

    def get_next_baselink_in_world_pose_estimate(self) -> CameraPoseSE3:
        return self.current_pose_estimate

"""


track
   if state == INIT
      keyframe = makekeyframe
      new_pose = 000
      transition to TRACKING
   elif state == TRACKING(sus=k)
      new_pose, matching_result = estimate pose by scan matching
      
      if matching_result == weird (bad matches)
         state.sus += 1
         if state.sus > limit:
            transition to LOST
      check how it's going
   elif state == LOST
      keyframe = makekeyframe
      new_pose = ?
"""


class FrontendState:
    class Init:
        """ Just initialized, need a keyframe. """
        ...

    @dataclass
    class Tracking:
        """ We are tracking, but maybe some iterations we have suspicions as not enough  """
        suspicion_level: int
        frames_since_keyframe: int
        keyframe: _Keyframe

    class Lost:
        """ Ooops! Doesn't look like anything else we have seen so far... """
        ...


@attr.s(auto_attribs=True)
class FrontendDebugData:
    """ Privileged information for the sake of debug """
    scene: List[RenderTriangle3d]


@attr.s(auto_attribs=True)
class FrontendTrackingResult:
    baselink_pose_estimate: TransformSE3


@attr.s(auto_attribs=True)
class Frontend:
    """ At this point, it just groups up stuff related to Frontend """
    matcher: OrbBasedFeatureMatcher
    cam_specs: CameraSpecs

    # stateful
    pose_tracker: VelocityPoseTracker
    keyframe: Optional[_Keyframe] = None
    state: FrontendState = attr.Factory(FrontendState.Init)
    debug_data: Optional[FrontendDebugData] = None

    @classmethod
    def from_defaults(
        cls,
        cam_specs: CameraSpecs,
        start_pose: Optional[CameraPoseSE3] = None,
        debug_data: Optional[FrontendDebugData] = None
    ):
        return cls(
            matcher=OrbBasedFeatureMatcher.build(),
            cam_specs=cam_specs,
            pose_tracker=VelocityPoseTracker.from_defaults() if start_pose is None else VelocityPoseTracker(start_pose),
            debug_data=debug_data
        )

    def _track(self, obs: Observation) -> Tuple[FrontendTrackingResult, FrontendState]:
        match self.state:
            case FrontendState.Init() | FrontendState.Lost():
                prior_baselink_pose_estimate = self.pose_tracker.get_next_baselink_in_world_pose_estimate()
                keyframe = estimate_keyframe(
                    obs=obs,
                    matcher=self.matcher,
                    baselink_pose=prior_baselink_pose_estimate,
                    cam_intrinsics=self.cam_specs.intrinsics,
                    cam_extrinsics=self.cam_specs.extrinsics,
                    debug_scene=self.debug_data.scene
                )

                resu = FrontendTrackingResult(baselink_pose_estimate=prior_baselink_pose_estimate)
                state = FrontendState.Tracking(
                    suspicion_level=0,
                    frames_since_keyframe=0,
                    keyframe=keyframe
                )
                return resu, state
            case FrontendState.Tracking(suspicion_level, frames_since_keyframe, keyframe):
                prior_baselink_pose_estimate = self.pose_tracker.get_next_baselink_in_world_pose_estimate()
                tracking_result = estimate_pose_wrt_keyframe(
                    obs=obs,
                    matcher=self.matcher,
                    cam_specs=self.cam_specs,
                    baselink_pose_estimate_in_world=prior_baselink_pose_estimate,   # TODO: oops
                    keyframe=keyframe
                )
                posterior_baselink_pose_estimate = tracking_result.pose_estimate
                # TODO: code that processes suspicion level & frames since last keyframe
                resu = FrontendTrackingResult(baselink_pose_estimate=posterior_baselink_pose_estimate)
                state = FrontendState.Tracking(
                    suspicion_level=0,
                    frames_since_keyframe=frames_since_keyframe+1,
                    keyframe=keyframe
                )
                return resu, state
            case _:
                    raise ValueError("Unhandled state", self.state)

    def track(self, obs: Observation) -> FrontendTrackingResult:
        result, state = self._track(obs)
        self.state = state
        self.pose_tracker.track(result.baselink_pose_estimate)
        return result


def run_couple_first_frames():
    dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-04-01--22-41-24.msgpack')
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    np.set_printoptions(suppress=True)  # TODO: remove

    localization_debugger = LocalizationDebugger.from_scene(
        scene=data_streamer.recorded_data.scene,
        cam_specs=data_streamer.get_cam_specs()
    )

    frontend = Frontend.from_defaults(
        cam_specs=data_streamer.get_cam_specs(),
        start_pose=get_SE3_pose(x=-2.5),
        debug_data=FrontendDebugData(scene=data_streamer.recorded_data.scene)
    )

    for i, obs in enumerate(data_streamer.stream()):
        frontend_resu = frontend.track(obs)

        print(i)
        print(f"est pose = {SE3_pose_to_xytheta(frontend_resu.baselink_pose_estimate).round(2)}")
        print(f"gt  pose = {SE3_pose_to_xytheta(obs.baselink_pose).round(2)}")

        localization_debugger.add_pose(frontend_resu.baselink_pose_estimate, color=BGRCuteColors.DARK_GRAY)
        localization_debugger.add_pose(obs.baselink_pose, color=BGRCuteColors.SKY_BLUE)
        img = localization_debugger.render()
        cv2.imshow('localization_debugger', img)
        cv2.waitKey(-1)


if __name__ == '__main__':
    run_couple_first_frames()

