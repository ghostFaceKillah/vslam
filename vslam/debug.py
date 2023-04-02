from typing import List, Dict, Optional, Iterable

import attr
import numpy as np

from plotting import Packer, Col, Row, Padding, TextRenderer
from sim.birds_eye_view_render import DisplayBirdseyeView, BirdseyeViewSpecifier
from sim.sim_types import CameraExtrinsics, RenderTriangle3d
from utils.colors import BGRCuteColors
from utils.custom_types import BGRImageArray
from utils.cv2_but_its_typed import cv2_circle
from utils.enum_utils import StrEnum
from utils.image import take_crop_around, magnify
from vslam.cam import CameraIntrinsics
from vslam.features import FeatureMatch
from vslam.transforms import px_2d_to_cam_coords_3d_homo, get_world_to_cam_coord_flip_matrix, homogenize
from vslam.types import CameraPoseSE3


class GeneralDebugPanes(StrEnum):
    DESC = 'desc'

class FeatureMatchDebugPanes(StrEnum):
    LEFT = 'left'
    RIGHT = 'right'
    LEFT_CROP = 'left_crop'
    RIGHT_CROP = 'right_crop'


@attr.define
class FeatureMatchDebugger:
    ui_layout: Packer
    soft_mark_matches_on_baseline_images: bool = False

    @classmethod
    def from_defaults(cls):
        layout = Col(
            Row(Padding("desc")),
            Row(
                Padding(FeatureMatchDebugPanes.LEFT_CROP),
                Padding(FeatureMatchDebugPanes.LEFT),
                Padding(FeatureMatchDebugPanes.RIGHT),
                Padding(FeatureMatchDebugPanes.RIGHT_CROP)
            ),
        )

        return cls(ui_layout=layout)

    def get_baseline_images(
            self,
            from_img: BGRImageArray,
            to_img: BGRImageArray,
            matches: List[FeatureMatch],
    ):
        # draw the matches
        from_canvas_img = np.copy(from_img)
        to_canvas_img = np.copy(to_img)

        if self.soft_mark_matches_on_baseline_images:
            for match in matches:
                cv2_circle(from_canvas_img, match.get_from_keypoint_px()[::-1], color=BGRCuteColors.GRASS_GREEN, radius=1,
                           thickness=1)
                cv2_circle(to_canvas_img, match.get_to_keypoint_px()[::-1], color=BGRCuteColors.GRASS_GREEN, radius=1,
                           thickness=1)

        return from_canvas_img, to_canvas_img

    def get_debug_image_dict_for_match(
        self,
        from_canvas_img,
        to_canvas_img,
        match
    ) -> Dict[FeatureMatchDebugPanes, BGRImageArray]:
        from_img = np.copy(from_canvas_img)
        to_img = np.copy(to_canvas_img)

        crop_from = take_crop_around(from_canvas_img, around_point=match.get_from_keypoint_px(), crop_size=(32, 32))
        crop_to = take_crop_around(to_canvas_img, around_point=match.get_to_keypoint_px(), crop_size=(32, 32))

        cv2_circle(from_img, match.get_from_keypoint_px()[::-1], color=BGRCuteColors.ORANGE, radius=10, thickness=4)
        cv2_circle(to_img, match.get_to_keypoint_px()[::-1], color=BGRCuteColors.ORANGE, radius=10, thickness=4)

        return {
            FeatureMatchDebugPanes.LEFT: from_img,
            FeatureMatchDebugPanes.RIGHT: to_img,
            FeatureMatchDebugPanes.LEFT_CROP: magnify(crop_from, factor=4.0),
            FeatureMatchDebugPanes.RIGHT_CROP: magnify(crop_to, factor=4.0),
        }

    def render(
        self,
        from_img: BGRImageArray,
        to_img: BGRImageArray,
        matches: List[FeatureMatch],
        depths: Optional[List[float]] = None
    ) -> Iterable[BGRImageArray]:

        assert depths is None or len(depths) == len(matches), f'{len(depths)=} != {len(matches)=}'
        from_canvas_img, to_canvas_img = self.get_baseline_images(from_img, to_img, matches)

        if depths is None:
            depth_txts = [''] * len(matches)
        else:
            depth_txts = ['diverged' if depth is None else f'depth: {depth:.2f}' for depth in depths]

        for i, (match, depth_txt) in enumerate(zip(matches, depth_txts)):

            name_to_image = self.get_debug_image_dict_for_match(from_canvas_img, to_canvas_img, match)

            desc = f"Match {i} out of {len(matches)}. Euc dist = {match.get_pixel_distance():.2f} " \
                   f"Hamming dist = {match.get_hamming_distance():.2f} " + depth_txt

            name_to_image['desc'] = TextRenderer().render(desc)

            img = self.ui_layout.render(name_to_image)

            yield img


class TriangulationDebugPanes(StrEnum):
    TRIANGULATION = 'triangulation'


@attr.define
class TriangulationDebugger:
    ui_layout: Packer
    feature_match_debugger: FeatureMatchDebugger    # helps us show the raw feature match

    @classmethod
    def from_defaults(cls):
        layout = Col(
            Row(Padding(GeneralDebugPanes.DESC)),
            Row(
                Padding(FeatureMatchDebugPanes.LEFT_CROP),
                Padding(FeatureMatchDebugPanes.LEFT),
                Padding(FeatureMatchDebugPanes.RIGHT),
                Padding(FeatureMatchDebugPanes.RIGHT_CROP)
            ),
            Row(Padding(TriangulationDebugPanes.TRIANGULATION))
        )

        return cls(
            ui_layout=layout,
            feature_match_debugger=FeatureMatchDebugger.from_defaults(),
        )

    def draw_triangulation_bird_eye_view(
            self,
            baselink_pose: CameraPoseSE3,
            match: FeatureMatch,
            camera_intrinsics: CameraIntrinsics,
            camera_extrinsics: CameraExtrinsics,
            triangles: List[RenderTriangle3d],
            depth_or_none: Optional[float]
    ) -> BGRImageArray:
        """
        Things to draw:
        1) [X] triangles, but make the color bleaker
        2) [X] view cones
        NO [N] 3) baselink as a point
        4) [ ] line from left eye's focal point to the right eye's feature
        5) [ ] line from right eye's focal point to the left eye's feature
        NO [ ] 6) line from baselink to intersection of the 2 above lines
        NO [ ] 7) point at the end of line with estimated depth ???
        8) [ ] point along left eye's line that is at estimated depth away from this eye's center
        """

        display_renderer = DisplayBirdseyeView.from_view_specifier(
            view_specifier=BirdseyeViewSpecifier.from_view_center(
                view_center=(baselink_pose[0, -1], baselink_pose[1, -1]),
                world_size=(60.0, 60.0)
            )
        )

        display_renderer.draw_triangles(triangles)

        display_renderer.draw_view_cone(
            at_pose=baselink_pose @ camera_extrinsics.get_pose_of_left_cam_in_baselink(),
            camera_intrinsics=camera_intrinsics,
            whiskers_thickness_px=1
        )
        display_renderer.draw_view_cone(
            at_pose=baselink_pose @ camera_extrinsics.get_pose_of_right_cam_in_baselink(),
            camera_intrinsics=camera_intrinsics,
            whiskers_thickness_px=1
        )

        # 4) [ ] line from left eye's focal point to the left eye's feature
        left_pose = baselink_pose @ camera_extrinsics.get_pose_of_left_cam_in_baselink()
        left_img_keypoint_px = match.get_from_keypoint_px()

        right_pose = baselink_pose @ camera_extrinsics.get_pose_of_right_cam_in_baselink()
        right_img_keypoint_px = match.get_to_keypoint_px()

        def draw_point(
                pose,
                keypoint_px
        ):
            keypoint_in_cam = homogenize(px_2d_to_cam_coords_3d_homo(np.array([keypoint_px]), camera_intrinsics)[0])
            world_in_flip = get_world_to_cam_coord_flip_matrix().T
            keypoint_in_cam_unflipped = world_in_flip @ keypoint_in_cam
            keypoint_in_cam_unflipped[-1] = 0   # it is direction, not a pose
            keypoint_in_world = pose @ keypoint_in_cam_unflipped

            start_3d = pose[:3, -1]   # write a function, comeon

            eff_depth = depth_or_none if depth_or_none is not None else 100.0

            end_3d = start_3d + eff_depth * keypoint_in_world[:3]

            start_2d = start_3d[:2]
            end_2d = end_3d[:2]

            display_renderer.draw_line(
                from_pt=start_2d,
                to_pt=end_2d,
                color=BGRCuteColors.DARK_BLUE
            )

        draw_point(left_pose, left_img_keypoint_px)
        draw_point(right_pose, right_img_keypoint_px)

        return display_renderer.get_image()

    def render(
        self,
        from_img: BGRImageArray,
        to_img: BGRImageArray,
        matches: List[FeatureMatch],
        depths: List[float],
        baselink_pose: CameraPoseSE3,
        camera_intrinsics: CameraIntrinsics,
        camera_extrinsics: CameraExtrinsics,
        triangles: List[RenderTriangle3d],
    ) -> Iterable[BGRImageArray]:

        assert depths is None or len(depths) == len(matches), f'{len(depths)=} != {len(matches)=}'
        from_canvas_img, to_canvas_img = self.feature_match_debugger.get_baseline_images(from_img, to_img, matches)

        if depths is None:
            depth_txts = [''] * len(matches)
        else:
            depth_txts = ['diverged' if depth is None else f'depth: {depth:.2f}' for depth in depths]

        for i, (match, depth_txt, depth) in enumerate(zip(matches, depth_txts, depths)):

            name_to_image = self.feature_match_debugger.get_debug_image_dict_for_match(from_canvas_img, to_canvas_img, match)

            desc = f"Match {i} out of {len(matches)}. Euc dist = {match.get_pixel_distance():.2f} " \
                   f"Hamming dist = {match.get_hamming_distance():.2f} " + depth_txt

            name_to_image[GeneralDebugPanes.DESC] = TextRenderer().render(desc)
            name_to_image[TriangulationDebugPanes.TRIANGULATION] = self.draw_triangulation_bird_eye_view(
                baselink_pose=baselink_pose,
                match=match,
                camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
                triangles=triangles,
                depth_or_none=depth
            )

            img = self.ui_layout.render(name_to_image)

            yield img





