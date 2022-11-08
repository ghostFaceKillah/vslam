import os
from custom_types import DirPath, FilePath


def get_im_path(
    dataset_path: DirPath = '/Users/misiu-dev/temp/phd/kitti-dataset/sequences',
        sequence_no: int = 0,
        cam_no: int = 0,
        image_no: int = 0
) -> FilePath:
    assert 0 <= sequence_no <= 21
    assert 0 <= cam_no <= 1

    return os.path.join(dataset_path, f"{sequence_no:02d}", f"image_{cam_no:d}", f"{image_no:06d}.png")


def read_calib_file(filepath):
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

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(data['P0'], (3, 4))
    P_rect_10 = np.reshape(data['P1'], (3, 4))
    # P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    # P_rect_30 = np.reshape(filedata['P3'], (3, 4))
    return data



if __name__ == '__main__':
    '/Users/misiu-dev/temp/phd/kitti-dataset/'
    '/Users/misiu-dev/temp/phd/kitti-dataset/sequences'

    # read the data
    # like 2 iamges
    # maybe triangulate it or sth
    print(get_im_path())

