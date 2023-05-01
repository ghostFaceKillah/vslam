import os

from defs import ROOT_DIR
from vslam.datasets.simdata import SimDataStreamer
from vslam.debug import LocalizationDebugger
from vslam.frontend import Frontend
from vslam.poses import get_SE3_pose
from vslam.running import ResultRecorder, run_slam_system

if __name__ == "__main__":
    dataset_path = os.path.join(ROOT_DIR, "data/short_recording_2023-04-20--22-46-06.msgpack")
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path, max_obs=130)

    run_slam_system(
        data_streamer=data_streamer,
        slam_system=Frontend.from_params(
            cam_specs=data_streamer.get_cam_specs(),
            start_pose=get_SE3_pose(y=-5.),
            scene_for_debug=data_streamer.recorded_data.scene,
            max_px_distance=200.,
            max_hamming_distance=128,
            keyframe_max_px_distance = 300.,
            keyframe_max_hamming_distance = 128,
            minimum_number_of_matches = 12,
            max_allowed_error = 0.05,
            outlier_rejection_margin = 0.01
    ),
        result_recorder=ResultRecorder(),
        localization_debugger_or_none=LocalizationDebugger.from_scene(
            scene=data_streamer.recorded_data.scene,
            cam_specs=data_streamer.get_cam_specs()
        )
    )

