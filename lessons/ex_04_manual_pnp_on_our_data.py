import os

import cv2
import numpy as np

from defs import ROOT_DIR
from sim.sim_types import Observation
from utils.custom_types import BGRImageArray
from vslam.datasets.simdata import SimDataStreamer
from vslam.debug import LocalizationDebugger
from vslam.frontend import FrontendStaticDebugData, Frontend, FrontendTrackingResult
from vslam.poses import get_SE3_pose, SE3_pose_to_xytheta


def _process_debug_info(
    iteration_number: int,
    frontend_resu: FrontendTrackingResult,
    obs: Observation,
    localization_debugger: LocalizationDebugger,
) -> BGRImageArray:
    # TODO: move all of this inside Localization debugger ?
    print(iteration_number)
    print(
        f"est pose = {SE3_pose_to_xytheta(frontend_resu.baselink_pose_estimate).round(2)}"
    )
    print(f"gt  pose = {SE3_pose_to_xytheta(obs.baselink_pose).round(2)}")

    if frontend_resu.debug_data.frames_since_keyframe == 0:
        localization_debugger.add_keyframe(
            keyframe_baselink_pose=frontend_resu.baselink_pose_estimate,
            keyframe_img=obs.left_eye_img,
        )

    localization_debugger.add_pose_estimate(
        baselink_pose_groundtruth=obs.baselink_pose,
        baselink_pose_estimate=frontend_resu.baselink_pose_estimate,
        current_left_eye_image=obs.left_eye_img,
        current_right_eye_image=obs.right_eye_img,
        feature_matches_or_none=frontend_resu.debug_data.feature_matches,
        frames_since_keyframe=frontend_resu.debug_data.frames_since_keyframe,
    )
    return localization_debugger.render()


def run_couple_first_frames():
    # dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-04-01--22-41-24.msgpack')   # short
    dataset_path = os.path.join(
        ROOT_DIR, "data/short_recording_2023-04-11--19-18-13.msgpack"
    )  # long
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    np.set_printoptions(suppress=True)  # TODO: remove

    localization_debugger = LocalizationDebugger.from_scene(
        scene=data_streamer.recorded_data.scene, cam_specs=data_streamer.get_cam_specs()
    )

    frontend = Frontend.from_defaults(
        cam_specs=data_streamer.get_cam_specs(),
        start_pose=get_SE3_pose(x=-2.5),
        debug_data=FrontendStaticDebugData(scene=data_streamer.recorded_data.scene),
    )

    for i, obs in enumerate(data_streamer.stream()):
        frontend_resu = frontend.track(obs)

        debug_img = _process_debug_info(i, frontend_resu, obs, localization_debugger)

        # TODO: just_show ?
        cv2.imshow("localization_debugger", debug_img)
        cv2.waitKey(-1)


if __name__ == "__main__":
    run_couple_first_frames()
