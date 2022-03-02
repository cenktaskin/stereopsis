from cv2 import cv2
import numpy as np

from calibration_setup import board_size
from data_io import get_img_from_dataset, save_camera_info, data_path, img_size

# some constants for functions below
# they represent corresponding values for different size/color images
win_scaling = {0: 2, 1: 2, 2: 7, 4: 7, 5: 2}
subpixel_win_size = {0: (7, 7), 1: (7, 7), 2: (3, 3), 4: (3, 3), 5: (7, 7)}
flags = {0: cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
         1: cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
         2: None,
         4: cv2.CALIB_CB_ADAPTIVE_THRESH,
         5: cv2.CALIB_CB_ADAPTIVE_THRESH
         }

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


def corner_finder(img_paths, cam_index, review=True):
    corner_wiki = {}
    i = 0
    for img_file in img_paths:
        img = get_img_from_dataset(img_file, cam_index)
        if cam_index < 2:  # then stereo (rgb) img
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        ret, corners = cv2.findChessboardCorners(image=gray_img, patternSize=board_size, flags=flags[cam_index])
        if ret:
            if camera_index != 4:  # in up upsampled img cornerSubPix makes corners worse
                corners = cv2.cornerSubPix(gray_img, corners, subpixel_win_size[cam_index], (-1, -1), criteria)
            if review:
                if not review_image(img, corners, ret, cam_index):
                    continue
            corner_wiki[img_file.stem[3:]] = corners

        i += 1
        print(f"\rFrames:{i}/{len(img_paths)} - found:{len(corner_wiki)}", end='')

    cv2.destroyAllWindows()
    print(f"\r{len(corner_wiki)} valid frames for calibration")
    return corner_wiki


def review_image(img, corners, ret, cam_index):
    if cam_index >= 2:
        gray_img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(img, board_size, corners, ret)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.resizeWindow("img", *np.array(img_size[cam_index]) * win_scaling[cam_index])
    while (key_press := cv2.waitKey() & 0xFF) not in [32, 8]:
        print(f"Wrong key is pressed, probably {chr(key_press)}")
        print(f"Try again...")
    return key_press == 32


if __name__ == "__main__":
    # Get the dataset
    dataset_name = '20220301'
    calibration_images_dir = data_path.joinpath('raw', f"calibration-{dataset_name}")
    calibration_images = sorted(calibration_images_dir.glob('st*'))

    camera_index = 5
    manual_review = True
    save_results = True

    corners_by_frame = corner_finder(calibration_images, camera_index, review=manual_review)

    if save_results and len(corners_by_frame.keys()) > 0:
        save_camera_info(corners_by_frame, camera_index, 'corners', dataset_name)
