"""
Pretty much reimplementation by hand of
# int Frontend::EstimateCurrentPose

Let us reimplement by hand, in an explicit way:
1) creation of the first keyframe
2) tracking of couple first frames after that

then we will do the full nice frontend implementation.
"""
import os

import attr
import numpy as np

from defs import ROOT_DIR
from sim.sim_types import Observation
from vslam.datasets.simdata import SimDataStreamer
from vslam.features import OrbBasedFeatureMatcher, OrbFeatureDetections
from vslam.pnp import gauss_netwon_pnp
from vslam.poses import get_SE3_pose
from vslam.transforms import px_2d_to_cam_coords_3d_homo, SE3_inverse
from vslam.triangulation import naive_triangulation
from vslam.types import WorldCoords3D, CameraPoseSE3, CameraIntrinsics


@attr.define
class _Keyframe:
    """ Draft class, will get changed"""
    pose: CameraPoseSE3

    points_3d_est: WorldCoords3D
    feature_detections: OrbFeatureDetections


def estimate_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        cam_intrinsics: CameraIntrinsics,
        left_cam_pose: CameraPoseSE3,
        pose_of_right_cam_in_left_cam: CameraPoseSE3,
):
    left_detections = matcher.detect(obs.left_eye_img)
    right_detections = matcher.detect(obs.right_eye_img)
    feature_matches = matcher.match(left_detections, right_detections)

    from_kp_px_coords_2d = np.array([fm.get_from_keypoint_px() for fm in feature_matches], dtype=np.int64)
    from_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(from_kp_px_coords_2d, cam_intrinsics)

    to_kp_px_coords_2d = np.array([fm.get_to_keypoint_px() for fm in feature_matches], dtype=np.int64)
    to_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(to_kp_px_coords_2d, cam_intrinsics)

    depths = naive_triangulation(
        points_in_cam_one=from_kp_cam_coords_3d_homo,
        points_in_cam_two=to_kp_cam_coords_3d_homo,
        cam_two_in_cam_one=pose_of_right_cam_in_left_cam
    )

    points_3d_est = []
    feature_descriptors = []
    keypoints = []

    assert len(depths) == len(from_kp_cam_coords_3d_homo) == len(feature_matches)

    for depth, point_homo, feature_match in zip(depths, from_kp_cam_coords_3d_homo, feature_matches):
        if depth is None:
            continue

        points_3d_est.append(point_homo * depth)
        feature_descriptors.append(feature_match.from_feature)
        keypoints.append(feature_match.from_keypoint)

    return _Keyframe(
        pose=left_cam_pose,
        points_3d_est=points_3d_est,
        feature_detections=OrbFeatureDetections(np.array(feature_descriptors), keypoints)
    )


@attr.define
class PoseTracker:
    # last_pose: CameraPoseSE3 = attr.Factory(get_SE3_pose)

    def track(self, pose: CameraPoseSE3):
        return pose


def estimate_pose_wrt_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        cam_intrinsics: CameraIntrinsics,
        left_cam_pose: CameraPoseSE3,
        keyframe: _Keyframe
):
    left_detections = matcher.detect(obs.left_eye_img)
    matches = matcher.match(left_detections, keyframe.feature_detections)

    # make matches debugger

    points_2d = []
    points_3d = []

    # resolve List[FeatureMatch] into 2d points that match a subset of keyframe.points_3d_est
    for match in matches:
        from_idx = match.raw_match.queryIdx
        to_idx = match.raw_match.trainIdx

        points_3d.append(keyframe.points_3d_est[to_idx])
        point_2d = np.array(match.get_to_keypoint_px()).astype(np.float64)
        points_2d.append(point_2d)

    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)

    # TODO: dimensions
    # move to function
    cam_ = (points_2d - np.array([cam_intrinsics.cy, cam_intrinsics.cx])) / np.array([cam_intrinsics.fx, cam_intrinsics.fy])

    points_3d_in_flipped_keyframe = points_3d_in_world @ SE3_inverse(keyframe_pose).T @ world_to_cam_flip
    gauss_netwon_pnp(
        inverse_of_camera_pose_initial_guess=SE3_inverse(left_cam_pose),
        points_3d_in_flipped_keyframe=,
        points_2d_in_img=cam_,
    )
    x = 1
    pass




def run_couple_first_frames():
    dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-02-04--17-08-25.msgpack')
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    cam_intrinsics = data_streamer.get_cam_intrinsics()
    matcher = OrbBasedFeatureMatcher.build()

    initial_cam_pose = get_SE3_pose(x=-2.5, y=-1.0)
    # y = -1, because this is left eye, and cam offset was 2
    # this doesn't matter, and could be assumed to be all 0, but alignment to world frame will make debugging easier

    # pose_tracker = PoseTracker()
    pose = initial_cam_pose

    pose_of_right_cam_in_left_cam = data_streamer.get_cam_specs().get_pose_of_right_cam_in_left_cam()

    for i, obs in enumerate(data_streamer.stream()):
        if i == 0:
            keyframe = estimate_keyframe(obs, matcher, cam_intrinsics, initial_cam_pose, pose_of_right_cam_in_left_cam)
        else:
            # TODO: probably need some kind of pose tracker ?
            estimate_pose_wrt_keyframe(
                obs,
                matcher,
                cam_intrinsics,
                pose,
                keyframe
            )
            ...





if __name__ == '__main__':
    run_couple_first_frames()

