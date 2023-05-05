import os

from defs import ROOT_DIR
from sim.actor_simulation import Simulation, PreRecordedActor, TriangleSceneRenderer
from vslam.datasets.simdata import SimDataStreamer
from vslam.debug import LocalizationDebugger
from vslam.frontend import Frontend
from vslam.running import ResultRecorder, run_slam_system


def run_full_frontend(live_data: bool = False):
    """
    Live data is slooooow, around 1 iteration per second on my M1 mac.
    Better get a recording, run `python -m sim.run` and put in the result path below
    """

    if live_data:
        data_streamer = Simulation.from_defaults(
            actor=PreRecordedActor.from_a_nice_trip(),
            scene_renderer=TriangleSceneRenderer.from_default(seed=42),
        )
    else:
        dataset_path = os.path.join(ROOT_DIR, "data/short_recording_2023-05-04--19-10-34.msgpack")
        data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path, max_obs=130)

    run_slam_system(
        data_streamer=data_streamer,
        slam_system=Frontend.from_params(
            cam_specs=data_streamer.get_cam_specs(),
            start_pose=data_streamer.get_initial_baselink_pose(),
            scene_for_debug=data_streamer.get_scene()
        ),
        result_recorder=ResultRecorder(),
        localization_debugger_or_none=LocalizationDebugger.from_scene(
            scene=data_streamer.get_scene(),
            cam_specs=data_streamer.get_cam_specs(),
        )
    )


if __name__ == '__main__':
    run_full_frontend(live_data=True)