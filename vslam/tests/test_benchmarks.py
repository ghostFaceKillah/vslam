from sim.actor_simulation import Simulation, PreRecordedActor, TriangleSceneRenderer
from vslam.frontend import Frontend
from vslam.running import ResultRecorder, run_slam_system


def test_benchmark():
    """ Test if system doesn't make too much error overall. """
    sim = Simulation.from_defaults(
        actor=PreRecordedActor.from_a_nice_trip(),
        scene_renderer=TriangleSceneRenderer.from_default(),
        max_iterations_or_none=130
    )

    metrics = run_slam_system(
        data_streamer=sim,
        slam_system=Frontend.from_params(
            cam_specs=sim.get_cam_specs(),
            start_pose=sim.initial_baselink_pose,
            scene_for_debug=sim.get_scene()
        ),
        result_recorder=ResultRecorder(),
        localization_debugger_or_none=None,
    )

    assert metrics.sum_euclidean_error < 12.
    assert metrics.sum_angular_error < 1.
    assert metrics.sum_euclidean_diff_error < 4.1
    assert metrics.sum_angular_diff_error < 0.15
