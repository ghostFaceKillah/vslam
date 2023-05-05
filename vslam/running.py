from typing import List, Optional

import attr
import numpy as np
import tqdm

from sim.sim_types import Observation
from utils.custom_types import BGRImageArray
from utils.image import just_show
from vslam.datasets.simdata import DataProvider
from vslam.debug import LocalizationDebugger
from vslam.frontend import FrontendTrackingResult, Frontend
from vslam.math import get_difference_of_angles
from vslam.poses import SE3_pose_to_xytheta
from vslam.types import Pose2DArray


@attr.define
class SlamPerformanceMetrics:
    sum_euclidean_error: float
    sum_angular_error: float
    sum_euclidean_diff_error: float
    sum_angular_diff_error: float


@attr.define
class ResultRecorder:
    est_poses: List[Pose2DArray] = attr.Factory(list)
    gt_poses: List[Pose2DArray] = attr.Factory(list)

    def record(
        self,
        obs: Observation,
        frontend_resu: FrontendTrackingResult
    ):
        est_pose = SE3_pose_to_xytheta(frontend_resu.baselink_pose_estimate)
        gt_pose = SE3_pose_to_xytheta(obs.baselink_pose)

        self.est_poses.append(est_pose)
        self.gt_poses.append(gt_pose)

    def emit_metrics(self) -> SlamPerformanceMetrics:
        est_poses = np.array(self.est_poses)
        gt_poses = np.array(self.gt_poses)

        gt_diffs = np.diff(gt_poses, axis=0)
        est_diffs = np.diff(est_poses, axis=0)

        return SlamPerformanceMetrics(
            sum_euclidean_diff_error = np.linalg.norm(gt_diffs[:, :2] - est_diffs[:, :2], axis=1).sum(),
            sum_angular_diff_error = np.abs(get_difference_of_angles(gt_diffs[:, 2], est_diffs[:, 2])).sum(),
            sum_euclidean_error = np.linalg.norm(gt_poses[:, :2] - est_poses[:, :2], axis=1).sum(),
            sum_angular_error = np.abs(get_difference_of_angles(gt_poses[:, 2], est_poses[:, 2])).sum()
        )


def _process_debug_info(
    iteration_number: int,
    frontend_resu: FrontendTrackingResult,
    obs: Observation,
    localization_debugger: LocalizationDebugger,
    verbose: bool = False
) -> BGRImageArray:
    # TODO: move all of this inside Localization debugger ?
    if frontend_resu.debug_data.frames_since_keyframe == 0:
        debug_feature_matches = frontend_resu.debug_data.keyframe_estimation_debug_data_or_none.relevant_feature_matches
    else:
        debug_feature_matches = frontend_resu.debug_data.keyframe_tracking_debug_data_or_none.all_feature_matches

    if frontend_resu.debug_data.frames_since_keyframe == 0:
        localization_debugger.add_keyframe(
            keyframe_baselink_pose=frontend_resu.baselink_pose_estimate,
            keyframe_left_img=obs.left_eye_img,
            keyframe_right_img=obs.right_eye_img,
            feature_matches_or_none=debug_feature_matches
        )

    if verbose:
        print(iteration_number)
        print(f"est pose = {SE3_pose_to_xytheta(frontend_resu.baselink_pose_estimate).round(2)}")
        print(f"gt  pose = {SE3_pose_to_xytheta(obs.baselink_pose).round(2)}")
        if frontend_resu.debug_data.frames_since_keyframe == 0:
            df = frontend_resu.debug_data.keyframe_estimation_debug_data_or_none.to_df()
            print("Keyframe estimation debug info")
            print(df.describe().round(2))
            print("correlations = ")
            print(df.dropna().corr())
        else:
            df = frontend_resu.debug_data.keyframe_tracking_debug_data_or_none.to_df()
            print("Keyframe tracking debug info")
            print(df.describe().round(4))
            print("correlations = ")
            print(df.corr())

    localization_debugger.add_pose_estimate(
        baselink_pose_groundtruth=obs.baselink_pose,
        baselink_pose_estimate=frontend_resu.baselink_pose_estimate,
        current_left_eye_image=obs.left_eye_img,
        current_right_eye_image=obs.right_eye_img,
        feature_matches_or_none=debug_feature_matches,
        frames_since_keyframe=frontend_resu.debug_data.frames_since_keyframe,
        current_frame_no=iteration_number
    )
    return localization_debugger.render()


def run_slam_system(
    data_streamer: DataProvider,
    slam_system: Frontend,
    result_recorder: ResultRecorder,
    localization_debugger_or_none: Optional[LocalizationDebugger]
) -> SlamPerformanceMetrics:

    for i, obs in tqdm.tqdm(enumerate(data_streamer.stream())):
        frontend_resu = slam_system.track(obs)
        result_recorder.record(obs, frontend_resu)

        if localization_debugger_or_none is not None:
            just_show(_process_debug_info(i, frontend_resu, obs, localization_debugger_or_none))

    return result_recorder.emit_metrics()
