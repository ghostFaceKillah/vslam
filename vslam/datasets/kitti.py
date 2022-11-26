import os

import attr
import numpy as np

from utils.custom_types import DirPath, FilePath


def get_im_path(
    dataset_path: DirPath = '/Users/misiu-dev/temp/phd/kitti-dataset/sequences',
    sequence_no: int = 0,
    cam_no: int = 0,
    image_no: int = 0
) -> FilePath:
    assert 0 <= sequence_no <= 21
    assert 0 <= cam_no <= 1

    return os.path.join(dataset_path, f"{sequence_no:02d}", f"image_{cam_no:d}", f"{image_no:06d}.png")


def get_calibration_path(
        dataset_path: DirPath = '/Users/misiu-dev/temp/phd/kitti-dataset/sequences',
        sequence_no: int = 0,
) -> FilePath:
    assert 0 <= sequence_no <= 21
    return os.path.join(dataset_path, f"{sequence_no:02d}", "calib.txt")


@attr.s(auto_attribs=True)
class KittiCalibration:
    camera_left_projection_matrix: np.ndarray['3,4', np.float64]
    camera_right_projection_matrix: np.ndarray['3,4', np.float64]


def read_calib_from_file(filepath: FilePath) -> KittiCalibration:
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
    # 10 is offset by 386.1448 mm toward the driver's side (it's negative in the x dimension)

    # I guess I should just place them directly in the constants
    # P_rect_20 = np.reshape(data['P2'], (3, 4))
    # P_rect_30 = np.reshape(data['P3'], (3, 4))

    # https://github.com/pratikac/kitti/blob/master/readme.raw.txt
    #  P_rect_xx: 3x4 projection matrix after rectification

    return KittiCalibration(
        camera_right_projection_matrix=P_rect_00,
        camera_left_projection_matrix=P_rect_10
    )