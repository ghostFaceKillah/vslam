import cv2
import numpy as np

from plotting import Col, Padding, Row, TextRenderer
from colors import BGRCuteColors
from utils.cv2_but_its_typed import cv2_circle
from utils.image import take_crop_around, magnify
from vslam.datasets.kitti import get_im_path, get_calibration_path, read_calib_from_file
from vslam.features import OrbBasedFeatureMatcher

if __name__ == '__main__':

    # read the data
    # like 2 iamges
    # maybe triangulate it or sth
    calibration = read_calib_from_file(get_calibration_path(sequence_no=3))

    sequence_no = 3
    im_left = cv2.imread(get_im_path(sequence_no=sequence_no, cam_no=1))
    im_right = cv2.imread(get_im_path(sequence_no=sequence_no, cam_no=0))

    matcher = OrbBasedFeatureMatcher.build()
    feature_matches = matcher.detect_and_match(im_left, im_right)

    print(f"Found {len(feature_matches)} feature matches")

    for i, match in enumerate(feature_matches):
        layout = Col(
            Row(Padding("desc")),
            Row(Padding('left_crop'), Padding('left')),
            Row(Padding('right_crop'), Padding('right')),
        )
        from_img = np.copy(im_left)
        to_img = np.copy(im_right)

        crop_from = take_crop_around(
            img=from_img,
            around_point=tuple(int(x) for x in match.from_keypoint.pt)[::-1],
            crop_size=(32, 32)
        )

        crop_to = take_crop_around(
            img=to_img,
            around_point=tuple(int(x) for x in match.to_keypoint.pt)[::-1],
            crop_size=(32, 32)
        )

        cv2_circle(
            image=from_img,
            center_coordinates=tuple(int(x) for x in match.from_keypoint.pt),
            color=BGRCuteColors.ORANGE,
            radius=10,
            thickness=4,
        )

        cv2_circle(
            image=to_img,
            center_coordinates=tuple(int(x) for x in match.to_keypoint.pt),
            color=BGRCuteColors.ORANGE,
            radius=10,
            thickness=4,
        )
        desc = f"Match {i} out of {len(feature_matches)}. " \
               f"Euc dist = {match.get_pixel_distance():.2f} " \
               f"Hamming dist = {match.get_hamming_distance():.2f}"

        img = layout.render({
            'desc': TextRenderer().render(desc),
            'left': from_img,
            'right': to_img,
            'left_crop': magnify(crop_from, factor=4.0),
            'right_crop': magnify(crop_to, factor=4.0),
        })

        cv2.imshow('wow', img)
        cv2.waitKey(-1)


