import time
from typing import List, Protocol

import attr
import cv2
import numpy as onp
from jax import numpy as np

from plotting import Col, Padding, Row, TextRenderer, Packer
from sim.birds_eye_view_render import get_view_spcifier_from_scene, render_birdseye_view, BirdseyeViewParams
from sim.egocentric_render import render_scene_pixelwise_depth
from sim.sample_scenes import get_triangles_in_sky_scene_2, get_two_triangle_scene
from sim.sim_types import RenderTriangle3d, CameraSpecs, Observation, Action, Recording
from sim.ui import key_to_maybe_transforms, InteractionTransforms
from utils.colors import BGRCuteColors
from utils.custom_types import BGRImageArray, BGRColor
from utils.image import magnify
from utils.profiling import just_time
from vslam.math import normalize_vector
from vslam.poses import get_SE3_pose
from vslam.types import CameraPoseSE3, Vector3d


@attr.define
class TriangleSceneRenderer:
    scene_triangles: List[RenderTriangle3d]
    birdseye_view_specifier: BirdseyeViewParams

    camera: CameraSpecs = attr.field(factory=CameraSpecs.from_default)

    light_direction_in_world: Vector3d = normalize_vector(np.array([1.0, -1.0, -8.0]))

    shade_color = BGRCuteColors.DARK_GRAY
    sky_color: BGRColor = BGRCuteColors.SKY_BLUE
    ground_color: BGRColor = tuple(x - 20 for x in BGRCuteColors.CYAN)

    @classmethod
    def from_easy_scene(cls):
        triangles = get_two_triangle_scene()
        return cls(scene_triangles=triangles, birdseye_view_specifier=get_view_spcifier_from_scene(triangles))

    @classmethod
    def from_default(cls):

        # triangles = get_two_triangle_scene()
        # triangles = get_cube_scene()
        # triangles = get_triangles_in_sky_scene()
        triangles = get_triangles_in_sky_scene_2()

        return cls(
            scene_triangles=triangles,
            birdseye_view_specifier=get_view_spcifier_from_scene(triangles),
        )

    def render_first_person_view(self, camera_pose: CameraPoseSE3) -> BGRImageArray:
        """ Renders the scene from the perspective of the camera """
        jax_array = render_scene_pixelwise_depth(
            self.camera.screen_h,
            self.camera.screen_w,
            camera_pose,
            self.scene_triangles,
            self.camera.cam_intrinsics,
            self.light_direction_in_world,
            self.sky_color,
            self.ground_color,
            self.shade_color,
            self.camera.clipping_surfaces
        )
        return onp.array(jax_array)

    def render_birdseye_view(self, camera_pose: CameraPoseSE3) -> BGRImageArray:
        """ Renders the scene from birdseye view """
        jax_array = render_birdseye_view(
            self.camera.screen_h,
            self.camera.screen_w,
            self.birdseye_view_specifier,
            camera_pose,
            self.camera.cam_intrinsics,
            self.scene_triangles,
            self.ground_color
        )
        return onp.array(jax_array)

    def render_left_eye(self, camera_pose: CameraPoseSE3) -> BGRImageArray:
        return self.render_first_person_view(camera_pose @ self.left_eye_offset())

    def render_right_eye(self, camera_pose: CameraPoseSE3) -> BGRImageArray:
        return self.render_first_person_view(camera_pose @ self.right_eye_offset())

    def left_eye_offset(self):
        return get_SE3_pose(y=-self.camera.distance_between_eyes / 2)

    def right_eye_offset(self):
        return get_SE3_pose(y=self.camera.distance_between_eyes / 2)


class Actor(Protocol):
    def act(self, obs: Observation) -> Action:
        ...


@attr.define
class ManualActor:
    layout: Packer
    text_renderer: TextRenderer = attr.field(factory=TextRenderer)

    @classmethod
    def from_default(cls):
        return cls(
            layout=Col(
                Row(Padding("desc")),
                Row(Padding('left_img'), Padding('right_img')),
                Row(Padding('birdseye_view')),
            )
        )

    def act(self, obs: Observation) -> Action:
        # reacts to images from the environment
        img = self.layout.render({
            'desc': self.text_renderer.render(f'frame {obs.frame_idx} pose {obs.camera_pose[:3, 3]}'),
            'left_img': obs.left_eye_img,
            'right_img': obs.right_eye_img,
            'birdseye_view': magnify(obs.bev_img, 0.5),
        })

        cv2.imshow('scene', onp.array(img))
        key = cv2.waitKey(-1)

        end = key == 27
        if end:
            print("caught escape key, exiting")

        return Action(
            transforms=key_to_maybe_transforms(key),
            end=key == ord('q')
        )


@attr.define
class Simulation:
    actor: Actor
    scene_renderer: TriangleSceneRenderer
    dt: float = 0.1

    def _get_obs(self, camera_pose: CameraPoseSE3, frame_idx: int, sim_time_s: float) -> Observation:
        """ Renders what eyes see and constructs observation object """
        with just_time('right eye render'):
            right_eye_screen = self.scene_renderer.render_first_person_view(camera_pose @ self.scene_renderer.right_eye_offset())
        with just_time('left eye render'):
            left_eye_screen = self.scene_renderer.render_first_person_view(camera_pose @ self.scene_renderer.left_eye_offset())

        with just_time('birdseye render'):
            bev_img = self.scene_renderer.render_birdseye_view(camera_pose)

        return Observation(
            left_eye_img=left_eye_screen,
            right_eye_img=right_eye_screen,
            bev_img=bev_img,
            camera_pose=camera_pose,
            frame_idx=frame_idx,
            timestamp=sim_time_s,
        )

    def simulate(
        self,
        initial_camera_pose: CameraPoseSE3 = get_SE3_pose(x=-2.5),   # looking toward +x direction in world frame, +z in camera
    ) -> Recording:
        """ Simulates the environment. """
        recorder = Recording(self.scene_renderer.camera)

        camera_pose = initial_camera_pose
        sim_time = time.time()

        i = 0
        action = Action.empty()

        while True:
            # mutates environment based on actions
            camera_pose = camera_pose @ action.transforms.camera

            if action.end:
                break

            obs = self._get_obs(camera_pose, i, sim_time)
            action = self.actor.act(obs)
            recorder.record_observation(obs)

            i += 1
            sim_time += self.dt

        return recorder


@attr.define
class PreRecordedActor(Actor):
    """ Plays back a pre-recorded sequence of actions"""
    actions: List[InteractionTransforms]
    idx: int = 0

    @classmethod
    def from_a_nice_trip(cls, short_trip: bool = True):
        """ Makes an agent that replays prerecorded actions and replays them. """

        if short_trip:
            actions = (
                [InteractionTransforms.go_straight()] * 20
                + [InteractionTransforms.turn_right(), InteractionTransforms.go_straight()] * 20
            )
        else:
            actions = (
                [InteractionTransforms.go_straight()] * 200
                + [InteractionTransforms.turn_right(), InteractionTransforms.go_straight()] * 45
                + [InteractionTransforms.go_straight()] * 40
                + [InteractionTransforms.turn_right(), InteractionTransforms.go_straight()] * 45
                + [InteractionTransforms.go_straight()] * 200
                + [InteractionTransforms.turn_right(), InteractionTransforms.go_straight()] * 45
                + [InteractionTransforms.go_straight()] * 250
                + [InteractionTransforms.turn_left(), InteractionTransforms.go_straight()] * (90 + 22)
                + [InteractionTransforms.go_straight()] * 300
            )

        return cls(actions=actions)

    def act(self, obs: Observation) -> Action:
        """ Plays back a pre-recorded sequence of actions"""
        if self.idx >= len(self.actions):
            return Action.done()
        else:
            self.idx += 1
            return Action(transforms=self.actions[self.idx - 1], end=False)




