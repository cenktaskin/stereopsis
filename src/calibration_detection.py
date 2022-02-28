from cv2 import cv2
import numpy as np

from calibration_setup import board_size, img_size
from dataio import get_img_from_dataset, save_camera_info, data_path

# some constants for functions below
# they represent corresponding values for different size/color images
win_scaling = {0: 2, 1: 2, 2: 5, 3: 2, 4: 2}
subpixel_win_size = {0: (11, 11), 1: (11, 11), 2: (4, 4), 3: (4, 4), 4: (11, 11)}
flags = {0: cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
         1: cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
         3: cv2.CALIB_CB_ADAPTIVE_THRESH,
         2: cv2.CALIB_CB_ADAPTIVE_THRESH,
         4: cv2.CALIB_CB_ADAPTIVE_THRESH}

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


def corner_finder(img_paths, cam_index, review=True):
    corner_wiki = {}
    for img_file in img_paths:
        img = get_img_from_dataset(img_file, cam_index)
        if cam_index < 2:  # then stereo (rgb) img
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        ret, corners = cv2.findChessboardCorners(image=gray_img, patternSize=board_size, flags=flags[cam_index])
        if ret:
            if camera_index < 4:  # in up upsampled img cornerSubPix makes corners worse
                corners = cv2.cornerSubPix(gray_img, corners, subpixel_win_size[cam_index], (-1, -1), criteria)
            if review:
                if not review_image(img, corners, ret, cam_index):
                    continue
            corner_wiki[img_file.stem[3:]] = corners

        print(f"\rValid frames:{len(corner_wiki)}/{len(img_paths)}", end='')

    cv2.destroyAllWindows()
    print(f"\r{len(corner_wiki)} valid frames for calibration")
    return corner_wiki


def review_image(img, corners, ret, cam_index, window_scale_factor=2):
    if cam_index >= 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(img, board_size, corners, ret)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.resizeWindow("img", *np.array(img_size[cam_index]) * window_scale_factor)
    while (key_press := cv2.waitKey() & 0xFF) not in [32, 8]:
        print(f"Wrong key is pressed, probably {chr(key_press)}")
        print(f"Try again...")
    return key_press == 32


if __name__ == "__main__":
    # Get the dataset
    calibration_images_dir = data_path.joinpath('raw', 'calibration-images-20210609')
    calibration_images = sorted(calibration_images_dir.glob('st*.jpeg'))

    camera_index = 2
    manual_review = True
    save_results = False

    corners_by_frame = corner_finder(calibration_images, camera_index, review=manual_review)

    if save_results:
        save_camera_info(corners_by_frame, camera_index, 'corners')
