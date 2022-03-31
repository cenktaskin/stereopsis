import matplotlib.pyplot as plt
from cv2 import cv2

from src.data_io import load_camera_info, data_path, get_img_from_dataset, upsample_ir_img, parse_stereo_img, \
    get_random_frame, img_size
import numpy as np
from random import sample
from src.calibration_setup import board_size

cam0_index = 0
cam1_index = 2
dataset = "20220301"

intrinsics0 = load_camera_info(cam0_index, 'intrinsics', dataset)
intrinsics1 = load_camera_info(cam1_index, 'intrinsics', dataset)
extrinsics1 = load_camera_info(cam1_index, 'extrinsics', dataset)

camera_matrix0 = intrinsics0['intrinsic_matrix']
camera_matrix1 = intrinsics1['intrinsic_matrix']
rot_wto0 = np.eye(3)
rot_wto2 = extrinsics1['rotation_matrix']
rot_2to0 = rot_wto0 @ np.linalg.inv(rot_wto2)
tra = extrinsics1['translation_vector']

corner_frames = load_camera_info(2, 'corners', dataset).keys()
for i in range(2):
    corner_frames = corner_frames & load_camera_info(i, 'corners', dataset).keys()

for tr in range(20):
    # get a frame with detected corners for debugging
    frame = sample(list(corner_frames), 1)[0]

    frame_path = data_path.joinpath('raw', f"calibration-{dataset}", f"st_{frame}.tiff")
    imgs = [get_img_from_dataset(frame_path, i) for i in range(4)]

    new_cmat0, _ = cv2.getOptimalNewCameraMatrix(camera_matrix0, intrinsics0['distortion_coeffs'], img_size[0], 0)
    undistorted0 = cv2.undistort(imgs[0], camera_matrix0, intrinsics0['distortion_coeffs'])
    gray_img = cv2.cvtColor(undistorted0, cv2.COLOR_BGR2GRAY)
    ret, rgb_corners = cv2.findChessboardCorners(image=gray_img, patternSize=board_size, flags=None)

    if not ret:
        cv2.imshow('failed',gray_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        continue

    new_cmat1, _ = cv2.getOptimalNewCameraMatrix(camera_matrix1, intrinsics1['distortion_coeffs'], img_size[2], 1)
    undistorted1 = cv2.undistort(imgs[2], camera_matrix1, intrinsics1['distortion_coeffs'], None, new_cmat1)
    gray_img = cv2.normalize(undistorted1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    ret, ir_corners = cv2.findChessboardCorners(image=gray_img, patternSize=board_size, flags=None)

    if not ret:
        cv2.imshow('failed',gray_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        continue

    undistorted_dp = cv2.undistort(imgs[3], camera_matrix1, intrinsics1['distortion_coeffs'], None, new_cmat1)

    corner_ir = ir_corners[0].flatten()
    corner_rgb = rgb_corners[0].flatten()

    p_ir = np.append(corner_ir, 1).reshape(3, -1)
    # https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    left_side = np.linalg.inv(rot_wto2) @ np.linalg.inv(camera_matrix1) @ p_ir
    right_side = np.linalg.inv(rot_wto2) @ tra

    point = np.round(corner_ir).astype(int)
    print(f"{point=}")
    depth_at_point = undistorted_dp[point[1], point[0]] * 10 ** 3
    s = depth_at_point + right_side[2] / left_side[2]
    new_p = np.linalg.inv(rot_wto2) @ (s * np.linalg.inv(camera_matrix1) @ p_ir - tra)

    img_pt = cv2.projectPoints(new_p, np.eye(3), np.zeros((3, 1)),
                               intrinsics0['intrinsic_matrix'], None)

    result = img_pt[0].flatten()
    print(f"{result=}")
    print(f"Real:{corner_rgb}")

    print(corner_ir)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(undistorted0[:, :, ::-1])
    plt.plot(*corner_rgb, marker='o', markersize=1, color="red")
    plt.plot(*result, marker='x', markersize=3, color="blue")
    plt.subplot(2, 2, 2)
    plt.imshow(imgs[0][:, :, ::-1])
    plt.plot(*corner_rgb, marker='o', markersize=1, color="red")
    plt.plot(*result, marker='x', markersize=3, color="blue")
    plt.subplot(2, 2, 3)
    plt.imshow(undistorted1, cmap='gray')
    plt.plot(*corner_ir, marker='o', markersize=1, color="red")
    plt.subplot(2, 2, 4)
    plt.imshow(undistorted_dp, cmap='gray')
    plt.plot(*corner_ir, marker='o', markersize=1, color="red")
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    plt.show()



