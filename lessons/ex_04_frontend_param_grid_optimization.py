import itertools
import os

import numpy as np
import pandas as pd
import tqdm

from defs import ROOT_DIR
from vslam.datasets.simdata import SimDataStreamer
from vslam.debug import LocalizationDebugger
from vslam.frontend import Frontend
from vslam.poses import get_SE3_pose
from vslam.running import ResultRecorder, run_slam_system, SlamPerformanceMetrics


def run_couple_first_frames(
    data_streamer: SimDataStreamer,
    max_px_distance=100.0,
    max_hamming_distance=31,
    keyframe_max_px_distance: float = np.inf,
    keyframe_max_hamming_distance: int = 31,
    minimum_number_of_matches=8,
    max_allowed_error=0.02,
    outlier_rejection_margin=0.01,
    show=False,
) -> SlamPerformanceMetrics:
    return run_slam_system(
        data_streamer=data_streamer,
        slam_system=Frontend.from_params(
            cam_specs=data_streamer.get_cam_specs(),
            start_pose=get_SE3_pose(y=-5.),
            scene_for_debug=data_streamer.recorded_data.scene,
            max_px_distance=max_px_distance,
            max_hamming_distance=max_hamming_distance,
            keyframe_max_px_distance=keyframe_max_px_distance,
            keyframe_max_hamming_distance=keyframe_max_hamming_distance,
            minimum_number_of_matches=minimum_number_of_matches,
            max_allowed_error=max_allowed_error,
            outlier_rejection_margin=outlier_rejection_margin
        ),
        result_recorder=ResultRecorder(),
        localization_debugger_or_none = LocalizationDebugger.from_scene(
            scene=data_streamer.recorded_data.scene,
            cam_specs=data_streamer.get_cam_specs()
        ) if show else None
    )


if __name__ == "__main__":
    # dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-04-01--22-41-24.msgpack')   # short
    dataset_path = os.path.join(
        # ROOT_DIR, "data/short_recording_2023-04-20--22-29-41.msgpack"    # short, many triangles, smooth
        # ROOT_DIR, "data/short_recording_2023-04-18--20-43-48.msgpack"  # classic, sparse, unsmooth turns
        ROOT_DIR, "data/short_recording_2023-04-20--22-46-06.msgpack"    # long, many triangles, smooth
    )  # long
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path, max_obs=130)

    np.set_printoptions(suppress=True)  # TODO: remove
    max_px_distance = [200.0, 400.0, float('inf')]
    max_hamming_distance = [64, 128, 256]
    keyframe_max_px_distance = [200.0, 300.0, float('inf')]
    keyframe_max_hamming_distance = [64, 128, 256]
    minimum_number_of_matches = [4, 8, 12, 16]
    max_allowed_error = [0.02, 0.05, 0.1]
    outlier_rejection_margin = [0.005, 0.01, 0.02]

    all_params = [
        params for params in itertools.product(
            max_px_distance,
            max_hamming_distance,
            keyframe_max_px_distance,
            keyframe_max_hamming_distance,
            minimum_number_of_matches,
            max_allowed_error,
            outlier_rejection_margin
        )
    ]

    metrics_agg = []
    best_euclidean_error_so_far = float('inf')

    for params in tqdm.tqdm(all_params):
        metrics = run_couple_first_frames(data_streamer, *params)
        if metrics.sum_euclidean_error < best_euclidean_error_so_far:
            best_euclidean_error_so_far = metrics.sum_euclidean_error
            print(f"{best_euclidean_error_so_far=:.3f}")
            print(params)
            print('- ' * 40)

        metrics_agg.append(metrics)

    params_again = list(map(list, zip(*all_params)))

    data = pd.DataFrame({
        "max_px_distance": params_again[0],
        "max_hamming_distance": params_again[1],
        "keyframe_max_px_distance": params_again[2],
        "keyframe_max_hamming_distance": params_again[3],
        "minimum_number_of_matches": params_again[4],
        "max_allowed_error": params_again[5],
        "outlier_rejection_margin": params_again[6],
        'euclidean_err': [metrics.sum_euclidean_error for metrics in metrics_agg],
        'angular_err': [metrics.sum_angular_error for metrics in metrics_agg],
        'euclidean_diff_err': [metrics.sum_euclidean_diff_error for metrics in metrics_agg],
        'angular_diff_err': [metrics.sum_angular_diff_error for metrics in metrics_agg],
    })

    data.to_csv('results.csv', index=False)
    print(data)


