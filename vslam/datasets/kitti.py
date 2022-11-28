import attr
import cv2
import numpy as np
import os

from utils.custom_types import DirPath, FilePath
from vslam.types import CameraPoseSE3, CameraIntrinsics


def _get_im_path(
    dataset_path: DirPath = '/Users/misiu-dev/temp/phd/kitti-dataset/sequences',
    sequence_no: int = 0,
    cam_no: int = 0,
    image_no: int = 0
) -> FilePath:
    assert 0 <= sequence_no <= 21
    assert 0 <= cam_no <= 1

    return os.path.join(dataset_path, f"{sequence_no:02d}", f"image_{cam_no:d}", f"{image_no:06d}.png")


def _get_calibration_path(
        dataset_path: DirPath = '/Users/misiu-dev/temp/phd/kitti-dataset/sequences',
        sequence_no: int = 0,
) -> FilePath:
    assert 0 <= sequence_no <= 21
    return os.path.join(dataset_path, f"{sequence_no:02d}", "calib.txt")


@attr.s(auto_attribs=True)
class KittiCalibration:
    camera_left_projection_matrix: CameraPoseSE3
    camera_right_projection_matrix: CameraPoseSE3


def _read_calib_from_file(filepath: FilePath) -> KittiCalibration:
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    P_rect_00 = np.reshape(data['P0'], (3, 4))  # these are the two relevant ones
    P_rect_10 = np.reshape(data['P1'], (3, 4))  # these are the two relevant ones
    # these are two identical cameras
    # cam 0 is the base
    # 10 is offset by 386.1448 pixels (lol) toward the driver's side (it's negative in the x dimension)
    # it's not mm as far as I understand

    # I guess I should just place them directly in the constants
    # P_rect_20 = np.reshape(data['P2'], (3, 4))
    # P_rect_30 = np.reshape(data['P3'], (3, 4))

    # https://github.com/pratikac/kitti/blob/master/readme.raw.txt
    #  P_rect_xx: 3x4 projection matrix after rectification.

    x = 1
    return KittiCalibration(
        camera_right_projection_matrix=P_rect_00,
        camera_left_projection_matrix=P_rect_10
    )


@attr.s(auto_attribs=True)
class KittiDataset:
    _sequence_no: int

    @classmethod
    def make(cls, sequence_no: int = 0):
        assert 0 <= sequence_no <= 21
        return cls(sequence_no=sequence_no )

    def get_left_image(self, image_no: int):
        return cv2.imread(_get_im_path(image_no=image_no, sequence_no=self._sequence_no, cam_no=1))

    def get_right_image(self, image_no: int):
        return cv2.imread(_get_im_path(image_no=image_no, sequence_no=self._sequence_no, cam_no=0))

    def get_calibration(self) -> KittiCalibration:
        return _read_calib_from_file(_get_calibration_path(sequence_no=self._sequence_no))

    def get_left_camera_intrinsics(self) -> CameraIntrinsics:
        """ Source: https://github.com/raulmur/ORB_SLAM2/blob/master/Examples/Stereo/KITTI04-12.yaml """
        # TODO: figure out this calibration, it looks all over the place
        return CameraIntrinsics(
            fx=707.0912,    # in pixels
            fy=707.0912,
            cx=601.8873,
            cy=183.1104
        )