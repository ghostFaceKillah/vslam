import itertools
import os

import numpy as np
import pandas as pd
import tqdm

from defs import ROOT_DIR
from sim.sim_types import Observation
from utils.custom_types import BGRImageArray
from vslam.datasets.simdata import SimDataStreamer
from vslam.debug import LocalizationDebugger
from vslam.features import OrbBasedFeatureMatcher
from vslam.frontend import FrontendStaticDebugData, Frontend, FrontendTrackingResult, FrontendPoseQualityEstimator
from vslam.poses import get_SE3_pose, SE3_pose_to_xytheta
from vslam.tracking import VelocityPoseTracker


def _process_debug_info(
    iteration_number: int,
    frontend_resu: FrontendTrackingResult,
    obs: Observation,
    localization_debugger: LocalizationDebugger,
) -> BGRImageArray:
    # TODO: move all of this inside Localization debugger ?
    print(iteration_number)
    print(f"est pose = {SE3_pose_to_xytheta(frontend_resu.baselink_pose_estimate).round(2)}")
    print(f"gt  pose = {SE3_pose_to_xytheta(obs.baselink_pose).round(2)}")

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
    )
    return localization_debugger.render()


def get_angular_error(est_theta, gt_theta):
    error = est_theta - gt_theta
    error = (error + np.pi) % (2 * np.pi) - np.pi
    return error


def run_couple_first_frames(
    data_streamer,
    max_px_distance=100.0,
    max_hamming_distance=31,
    minimum_number_of_matches=8,
    max_allowed_error=0.02
):
    localization_debugger = LocalizationDebugger.from_scene(scene=data_streamer.recorded_data.scene, cam_specs=data_streamer.get_cam_specs())

    frontend = Frontend(
        matcher=OrbBasedFeatureMatcher.build(
            max_px_distance=max_px_distance,
            max_hamming_distance=max_hamming_distance,
        ),
        cam_specs=data_streamer.get_cam_specs(),
        pose_tracker=VelocityPoseTracker(get_SE3_pose(x=-2.5)),
        tracking_quality_estimator=FrontendPoseQualityEstimator(
            minimum_number_of_matches=minimum_number_of_matches,
            max_allowed_error=max_allowed_error,
        ),
        debug_data=FrontendStaticDebugData(scene=data_streamer.recorded_data.scene),
    )

    sum_euclidean_error = 0.0
    sum_angular_error = 0.0

    for i, obs in enumerate(data_streamer.stream()):
        frontend_resu = frontend.track(obs)
        est_pose = SE3_pose_to_xytheta(frontend_resu.baselink_pose_estimate)
        gt_pose = SE3_pose_to_xytheta(obs.baselink_pose)

        euclidean_err = np.linalg.norm(est_pose[:2] - gt_pose[:2])
        angular_err = np.abs(get_angular_error(est_pose[2], gt_pose[2]))

        sum_euclidean_error += euclidean_err
        sum_angular_error += angular_err

        # print(f"{i}")
        # print(f"est pose = {est_pose.round(2)}")
        # print(f"gt  pose = {gt_pose.round(2)}")
        # print(f"Euclidean error: {euclidean_err:.2f}")
        # print(f"Angular error: {angular_err:.2f} radians")

        # img = _process_debug_info(i, frontend_resu, obs, localization_debugger)
        # cv2.imshow('eheh', img)
        # cv2.waitKey(-1)

        # if i > 80:
        #     break

    # print(f'sum angular error = {sum_angular_error:.2f} sum_euclidean_eror = {sum_euclidean_error:.2f}')
    return sum_euclidean_error, sum_angular_error


if __name__ == "__main__":
    # dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-04-01--22-41-24.msgpack')   # short
    dataset_path = os.path.join(
        ROOT_DIR, "data/short_recording_2023-04-18--19-19-43.msgpack"
        # ROOT_DIR, "data/short_recording_2023-04-18--20-43-48.msgpack"
    )  # long
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    run_couple_first_frames(
            data_streamer,
            max_px_distance=150.0,
            max_hamming_distance=32,
            minimum_number_of_matches=12,
            max_allowed_error=0.01
    )

    np.set_printoptions(suppress=True)  # TODO: remove
    max_px_distance = [50.0, 100.0, 150.0, 200.0, 400.0]
    max_hamming_distance = [4, 8, 16, 32, 64]
    minimum_number_of_matches = [4, 8, 12, 16]
    max_allowed_error = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]

    all_params = [
        params for params in itertools.product(max_px_distance, max_hamming_distance, minimum_number_of_matches, max_allowed_error)
    ]

    es_, as_ = [], []

    for params in tqdm.tqdm(all_params):
        e, a = run_couple_first_frames(data_streamer, *params)
        es_.append(np.round(e, 2))
        as_.append(np.round(a, 2))

    params_again = list(map(list, zip(*all_params)))

    data = pd.DataFrame({
        "max_px_distance": params_again[0],
        "max_hamming_distance": params_again[1],
        "minimum_number_of_matches": params_again[2],
        "max_allowed_error": params_again[3],
        'euclidean_err': es_,
        'angular_err': as_,
    })

    data.to_csv('results.csv')
    print(data)


