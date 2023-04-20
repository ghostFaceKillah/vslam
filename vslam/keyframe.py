from typing import Optional, List, Tuple

import attr
import cv2
import numpy as np

from sim.sim_types import Observation, CameraExtrinsics, RenderTriangle3d, CameraSpecs
from utils.custom_types import BGRImageArray
from vslam.cam import CameraIntrinsics
from vslam.debug import FeatureMatchDebugger, TriangulationDebugger
from vslam.features import OrbFeatureDetections, OrbBasedFeatureMatcher, FeatureMatch
from vslam.pnp import GaussNetwonAuxillaryInfo, gauss_netwon_pnp
from vslam.poses import correct_SE3_matrix_inplace
from vslam.transforms import px_2d_to_cam_coords_3d_homo, dehomogenize, CAM_TO_WORLD_FLIP, homogenize, SE3_inverse, \
    px_2d_to_img_coords_2d
from vslam.triangulation import naive_triangulation, DepthEstimate
from vslam.types import CameraPoseSE3, WorldCoords3D, TransformSE3


@attr.define
class Keyframe:
    """ Keyframe: We will match features to features on this image and
    estimate pose wrt to coordinate system of this image . """
    pose: CameraPoseSE3
    image: BGRImageArray
    points_3d_est: WorldCoords3D
    feature_detections: OrbFeatureDetections


@attr.define
class KeyFrameEstimationDebugData:
    relevant_feature_matches: List[FeatureMatch]   # feature matches with succesfully estimated depth
    all_feature_matches: List[FeatureMatch]
    all_depth_estimates: List[DepthEstimate]


def estimate_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        baselink_pose: CameraPoseSE3,
        cam_intrinsics: CameraIntrinsics,
        cam_extrinsics: CameraExtrinsics,
        debug_feature_matches: bool = False,
        debug_depth_estimation: bool = False,
        debug_scene: Optional[List[RenderTriangle3d]] = None,   # purely for debug vis of depth
) -> Tuple[Keyframe, KeyFrameEstimationDebugData]:
    left_cam_pose = baselink_pose @ cam_extrinsics.get_pose_of_left_cam_in_baselink()

    left_detections = matcher.detect(obs.left_eye_img)
    right_detections = matcher.detect(obs.right_eye_img)
    feature_matches = matcher.match(left_detections, right_detections)

    # def estimate depth from feature matches
    from_kp_px_coords_2d = np.array([fm.get_from_keypoint_px() for fm in feature_matches], dtype=np.int64)
    from_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(from_kp_px_coords_2d, cam_intrinsics)

    to_kp_px_coords_2d = np.array([fm.get_to_keypoint_px() for fm in feature_matches], dtype=np.int64)
    to_kp_cam_coords_3d_homo = px_2d_to_cam_coords_3d_homo(to_kp_px_coords_2d, cam_intrinsics)

    depth_estimates = naive_triangulation(
        points_in_cam_one=from_kp_cam_coords_3d_homo,
        points_in_cam_two=to_kp_cam_coords_3d_homo,
        cam_one_in_two=cam_extrinsics.get_pose_of_left_cam_in_right_cam()
    )

    depths = [est.depth_or_none for est in depth_estimates]

    if debug_feature_matches:
        debugger = FeatureMatchDebugger.from_defaults()

        for img in debugger.render(obs.left_eye_img, obs.right_eye_img, feature_matches, depths):
            cv2.imshow('wow', img)
            cv2.waitKey(-1)

    points_3d_est = []
    feature_descriptors = []
    keypoints = []
    relevant_feature_matches = []

    assert len(depths) == len(from_kp_cam_coords_3d_homo) == len(feature_matches)

    for depth_estimate, point_homo, feature_match in zip(depth_estimates, from_kp_cam_coords_3d_homo, feature_matches):
        if depth_estimate.depth_or_none is None:
            continue

        relevant_feature_matches.append(feature_match)
        points_3d_est.append(dehomogenize(CAM_TO_WORLD_FLIP @ homogenize(point_homo * depth_estimate.depth_or_none)))
        feature_descriptors.append(feature_match.from_feature)
        keypoints.append(feature_match.from_keypoint)

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

    debug_data = KeyFrameEstimationDebugData(
        relevant_feature_matches=relevant_feature_matches,
        all_feature_matches=feature_matches,
        all_depth_estimates=depth_estimates
    )

    keyframe = Keyframe(
        image=obs.left_eye_img,
        pose=left_cam_pose,
        points_3d_est=points_3d_est,
        feature_detections=OrbFeatureDetections(np.array(feature_descriptors), keypoints)
    )

    return keyframe, debug_data


class KeyframeMatchPoseTrackingResult:
    @attr.define
    class Failure:
        reason: str

    @attr.define
    class Success:
        pose_estimate: CameraPoseSE3
        tracking_quality_info: GaussNetwonAuxillaryInfo   # mildly bad design to entangle this to Gauss Newton specifically


@attr.s(auto_attribs=True)
class KeyframeTrackingDebugData:
    all_feature_matches: List[FeatureMatch]
    reprojection_errors_or_none: Optional[List[float]]


def estimate_pose_wrt_keyframe(
        obs: Observation,
        matcher: OrbBasedFeatureMatcher,
        cam_specs: CameraSpecs,
        baselink_pose_estimate_in_world: TransformSE3,
        keyframe: Keyframe,
        min_no_matches_needed: int = 5,
        debug_feature_matches: bool = False
) -> Tuple[KeyframeMatchPoseTrackingResult, KeyframeTrackingDebugData]:
    left_detections = matcher.detect(obs.left_eye_img)
    matches = matcher.match(keyframe.feature_detections, left_detections)

    if len(matches) < min_no_matches_needed:
        debug_data = KeyframeTrackingDebugData(
            all_feature_matches=matches,
            reprojection_errors_or_none=None
        )
        result = KeyframeMatchPoseTrackingResult.Failure(reason=f'not enough matches {len(matches)=}')
        return result, debug_data

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

    # what happens if there are no matches ?
    points_3d = np.array(points_3d)

    left_camera_pose_in_world = correct_SE3_matrix_inplace(
        baselink_pose_estimate_in_world @ cam_specs.extrinsics.get_pose_of_left_cam_in_baselink()
    )
    camera_pose_guess_in_keyframe = correct_SE3_matrix_inplace(SE3_inverse(keyframe.pose) @ left_camera_pose_in_world)

    posterior_left_cam_pose_estimate_in_keyframe, gauss_newton_info = gauss_netwon_pnp(
        camera_pose_initial_guess_in_keyframe=camera_pose_guess_in_keyframe,
        points_3d_in_keyframe=homogenize(points_3d),
        points_2d_in_img=px_2d_to_img_coords_2d(np.array(points_2d), cam_specs.intrinsics),
        verbose=False
    )

    # want: world T baselink
    posterior_baselink_pose_estimate_in_world = correct_SE3_matrix_inplace(
        keyframe.pose   # world T keyframe
        @ posterior_left_cam_pose_estimate_in_keyframe  # # keyframe T cam
        @ SE3_inverse(cam_specs.extrinsics.get_pose_of_left_cam_in_baselink())   # cam T baselink
    )

    resu = KeyframeMatchPoseTrackingResult.Success(
        posterior_baselink_pose_estimate_in_world,
        tracking_quality_info=gauss_newton_info
    )

    debug_data = KeyframeTrackingDebugData(
        all_feature_matches=matches,
        reprojection_errors_or_none=gauss_newton_info.euclidean_errors
    )

    return resu, debug_data
