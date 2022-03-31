import matplotlib.pyplot as plt
from cv2 import cv2

from data_io import load_camera_info, data_path, get_img_from_dataset, upsample_ir_img, parse_stereo_img, \
    get_random_frame, img_size
import numpy as np
from random import sample
from calibration_setup import board_size

cam0_index = 0
cam1_index = 2
dataset = "20220301"

intrinsics0 = load_camera_info(cam0_index, 'intrinsics', dataset)
intrinsics1 = load_camera_info(cam1_index, 'intrinsics', dataset)
extrinsics1 = load_camera_info(cam1_index, 'extrinsics', dataset)


## undistorted or distorted mapping comparison

for tr in range(20):
    # get a frame with detected corners for debugging
    corner_frames = load_camera_info(2, 'corners', dataset).keys()
    for i in range(2):
        corner_frames = corner_frames & load_camera_info(i, 'corners', dataset).keys()
    frame = sample(list(corner_frames), 1)[0]

    frame_path = data_path.joinpath('raw', f"calibration-{dataset}", f"st_{frame}.tiff")
    imgs = [get_img_from_dataset(frame_path, i) for i in range(4)]

    the_corners = [load_camera_info(i, 'corners', dataset)[frame][0].flatten() for i in range(3)]
    the_corners.append(the_corners[-1])  # copy the point from ir to depth

    camera_matrix0 = intrinsics0['intrinsic_matrix']
    camera_matrix1 = intrinsics1['intrinsic_matrix']
    rot_wto0 = np.eye(3)
    rot_wto2 = extrinsics1['rotation_matrix']
    rot_2to0 = rot_wto0 @ np.linalg.inv(rot_wto2)
    tra = extrinsics1['translation_vector']

    corner_ir = the_corners[2]

    # undistort find corner try again
    print(f"{corner_ir=}")
    new_cmat, _ = cv2.getOptimalNewCameraMatrix(camera_matrix1, intrinsics1['distortion_coeffs'], img_size[2], 1)
    undist_p = cv2.undistortPoints(corner_ir, camera_matrix1, intrinsics1['distortion_coeffs'], R=rot_wto2,
                                   P=new_cmat)
    print(f"{undist_p=}")
    undistorted_ir = cv2.undistort(imgs[2], camera_matrix1, intrinsics1['distortion_coeffs'], None, new_cmat)
    gray_img = cv2.normalize(undistorted_ir, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    ret, und_cor = cv2.findChessboardCorners(image=gray_img, patternSize=board_size, flags=None)
    if ret and False:
        undistorted_corner = und_cor[0].flatten()
        plt.subplot(1, 2, 1)
        plt.imshow(imgs[2], cmap='gray')
        plt.plot(*undist_p.flatten(), marker='x', markersize=3, color="blue")
        plt.plot(*undistorted_corner, marker='x', markersize=3, color="cyan")
        plt.plot(*corner_ir, marker='o', markersize=1, color="red")
        plt.subplot(1, 2, 2)
        plt.imshow(undistorted_ir, cmap='gray')
        plt.plot(*undist_p.flatten(), marker='x', markersize=3, color="blue")
        plt.plot(*undistorted_corner, marker='x', markersize=3, color="cyan")
        plt.plot(*corner_ir, marker='o', markersize=1, color="red")
        figManager = plt.get_current_fig_manager()
        figManager.resize(*figManager.window.maxsize())
        plt.show()

    p_ir = np.append(corner_ir, 1).reshape(3, -1)
    # via undostprtion
    #p_ir = np.append(undist_p, 1).reshape(3, -1)
    # https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    left_side = np.linalg.inv(rot_wto2) @ np.linalg.inv(camera_matrix1) @ p_ir
    right_side = np.linalg.inv(rot_wto2) @ tra

    point = np.round(corner_ir).astype(int)
    print(f"{point=}")
    depth_at_point = imgs[3][point[1], point[0]] * 10 ** 3
    s = depth_at_point + right_side[2] / left_side[2]
    new_p = np.linalg.inv(rot_wto2) @ (s * np.linalg.inv(camera_matrix1) @ p_ir - tra)

    img_pt = cv2.projectPoints(new_p, np.eye(3), np.zeros((3, 1)),
                               intrinsics0['intrinsic_matrix'], intrinsics0['distortion_coeffs'])

    result = img_pt[0].flatten()
    print(f"{result=}")
    print(f"Real:{the_corners[0]}")

    # via undostprtion
    p_ir = np.append(undist_p, 1).reshape(3, -1)
    # https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    left_side = np.linalg.inv(rot_wto2) @ np.linalg.inv(new_cmat) @ p_ir
    right_side = np.linalg.inv(rot_wto2) @ tra

    point = np.round(corner_ir).astype(int)
    print(f"{point=}")
    depth_at_point = imgs[3][point[1], point[0]] * 10 ** 3
    s = depth_at_point + right_side[2] / left_side[2]
    new_p = np.linalg.inv(rot_wto2) @ (s * np.linalg.inv(new_cmat) @ p_ir - tra)

    img_pt = cv2.projectPoints(new_p, np.eye(3), np.zeros((3, 1)),
                               intrinsics0['intrinsic_matrix'], intrinsics0['distortion_coeffs'])

    result2 = img_pt[0].flatten()


    imgs[3] = undistorted_ir
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if len(imgs[i].shape) == 3:
            plt.imshow(imgs[i][:, :, ::-1])
        else:
            plt.imshow(imgs[i], cmap='gray')
        plt.plot(*the_corners[i], marker='o', markersize=3, color="red")

        if i == 0:
            plt.plot(*result, marker='x', markersize=3, color="blue")
            plt.plot(*result2, marker='x', markersize=3, color="green")
        if i >= 2:
            plt.plot(*undist_p.flatten(), marker='x', markersize=3, color='red')
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())

    plt.show()
