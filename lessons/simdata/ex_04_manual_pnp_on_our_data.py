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
import cv2
import numpy as np

from typing import Optional

from defs import ROOT_DIR
from sim.sample_scenes import get_triangles_in_sky_scene_2
from sim.sim_types import Observation, CameraSpecs
from utils.custom_types import BGRImageArray
from vslam.datasets.simdata import SimDataStreamer
from vslam.features import OrbBasedFeatureMatcher, OrbFeatureDetections, FeatureMatchDebugger
from vslam.pnp import gauss_netwon_pnp
from vslam.poses import get_SE3_pose, SE3_pose_to_xytheta
from vslam.transforms import px_2d_to_cam_coords_3d_homo, SE3_inverse, px_2d_to_img_coords_2d, \
    get_world_to_cam_coord_flip_matrix, homogenize, CAM_TO_WORLD_FLIP, dehomogenize
from vslam.triangulation import naive_triangulation
from vslam.types import WorldCoords3D, CameraPoseSE3, CameraIntrinsics, TransformSE3


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
        cam_intrinsics: CameraIntrinsics,
        left_cam_pose: CameraPoseSE3,
        pose_of_right_cam_in_left_cam: CameraPoseSE3,
        debug_feature_matches: bool = False
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


def estimate_pose_wrt_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        cam_intrinsics: CameraIntrinsics,
        camera_pose_guess_in_world: CameraPoseSE3,
        keyframe: _Keyframe,
        debug_feature_matches: bool = False
):
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
    """
    camera_pose_initial_guess: Optional[CameraPoseSE3],     # initial guess has to be relative to keyframe!
    points_3d_in_keyframe: WorldCoords3D,   # if those are in keyframe, the pose estimate will be relative to keyframe
    points_2d_in_img: ImgCoords2d,
    """

    camera_pose_guess_in_keyframe = SE3_inverse(keyframe.pose) @ camera_pose_guess_in_world

    new_pose_estimate = gauss_netwon_pnp(
        camera_pose_initial_guess_in_keyframe=camera_pose_guess_in_keyframe,
        points_3d_in_keyframe=homogenize(points_3d),
        points_2d_in_img=px_2d_to_img_coords_2d(np.array(points_2d), cam_intrinsics),
        verbose=False
    )

    return new_pose_estimate


def fake_estimate_pose_wrt_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        cam_intrinsics: CameraIntrinsics,
        world_to_cam_flip: TransformSE3,
        camera_pose_guess_in_world: CameraPoseSE3,
        keyframe: _Keyframe,
        debug_feature_matches: bool = False

):
    """ Use access to priveliged data to diagnose the relative pose estimation """
    # compute 3d points in keyframe
    # compute 2d points in new camera pose


    x = 1

    pass




class VelocityPoseTracker:
    # it predicts the next pose
    current_pose_estimate: CameraPoseSE3

    def track(self, new_pose: CameraPoseSE3):
        # will
        self.current_pose_estimate = new_pose
        pass

    def get_next_baselink_in_world_pose_estimate(self) -> CameraPoseSE3:
        ...


@attr.s(auto_attribs=True)
class FrontEnd:
    """ At this point, it just groups up stuff related to Frontend """
    matcher: OrbBasedFeatureMatcher
    cam_specs: CameraSpecs

    # stateful
    pose_tracker: VelocityPoseTracker
    keyframe: Optional[_Keyframe] = None

    def track(self, obs: Observation) -> CameraPoseSE3:
        if self.keyframe is None:
            pose_guess = self.pose_tracker.get_next_baselink_in_world_pose_estimate()
            keyframe = estimate_keyframe(
                obs=obs,
                matcher=self.matcher,
                cam_intrinsics=self.cam_specs.cam_intrinsics,
                left_cam_pose=self.pose_tracker.get_next_baselink_in_world_pose_estimate() @ self.cam_specs.get_pose_of_left_cam_in_baselink(),
                pose_of_right_cam_in_left_cam=self.cam_specs.get_pose_of_right_cam_in_left_cam()
            )
            ...
            # ?
        else:
            ...

def run_couple_first_frames():
    # dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-02-26--13-41-16.msgpack')
    dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-02-28--08-45-26.msgpack')
    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    cam_intrinsics = data_streamer.get_cam_intrinsics()
    matcher = OrbBasedFeatureMatcher.build()

    initial_cam_pose = get_SE3_pose(x=-2.5, y=-1.0)
    # y = -1, because this is left eye, and cam offset was 2
    # this doesn't matter, and could be assumed to be all 0, but alignment to world frame will make debugging easier

    # pose_tracker = PoseTracker()
    pose = initial_cam_pose
    pose_of_left_cam_in_baselink = data_streamer.get_cam_specs().get_pose_of_left_cam_in_baselink()

    for i, obs in enumerate(data_streamer.stream()):
        # off by one error I think
        if i == 0:
            keyframe = estimate_keyframe(
                obs=obs,
                matcher=matcher,
                cam_intrinsics=cam_intrinsics,
                left_cam_pose=initial_cam_pose,
                pose_of_right_cam_in_left_cam= data_streamer.get_cam_specs().get_pose_of_right_cam_in_left_cam()
            )
            x = 1
        else:
            # TODO: probably need some kind of pose tracker ?
            new_pose_estimate = estimate_pose_wrt_keyframe(
                obs=obs,
                matcher=matcher,
                cam_intrinsics=cam_intrinsics,
                camera_pose_guess_in_world=pose,
                keyframe=keyframe
            )
            np.set_printoptions(suppress=True)
            print(i)
            # print(f"{new_pose_estimate.round(2)=}")
            # print(f"{(keyframe.pose @ new_pose_estimate).round(2)=}")
            # print(f"{obs.baselink_pose=}")
            print(f"est pose = {SE3_pose_to_xytheta(keyframe.pose @ new_pose_estimate).round(2)}")
            print(f"gt  pose = {SE3_pose_to_xytheta(obs.baselink_pose @ pose_of_left_cam_in_baselink).round(2)}")

            pose = keyframe.pose @ new_pose_estimate
            x = 1


if __name__ == '__main__':
    run_couple_first_frames()

