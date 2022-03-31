from cv2 import cv2
import matplotlib.pyplot as plt
from data_io import load_camera_info, get_img_from_dataset, data_path, img_size
from pathlib import Path
import numpy as np
from random import sample


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        print(tuple(pt1[0]))
        img1 = cv2.circle(img1, tuple(np.round(pt1[0]).astype(int)), 1, color, -1)
        img2 = cv2.circle(img2, tuple(np.round(pt2[0]).astype(int)), 5, color, -1)
    return img1, img2


cam0_index = 0
cam1_index = 2

dataset = "20220301"
corners0 = load_camera_info(cam0_index, 'corners', dataset)
corners1 = load_camera_info(cam1_index, 'corners', dataset)
intrinsics0 = load_camera_info(cam0_index, 'intrinsics', dataset)
intrinsics1 = load_camera_info(cam1_index, 'intrinsics', dataset)
extrinsics1 = load_camera_info(cam1_index, 'extrinsics', dataset)

common_frames = corners0.keys() & corners1.keys()
sample_index = sample(list(common_frames), 1)[0]
sample_path = data_path.joinpath("raw", f"calibration-{dataset}", f"st_{sample_index}.tiff")
rgb_img = get_img_from_dataset(sample_path, cam0_index)
ir_img = get_img_from_dataset(sample_path, cam1_index)

fund_mat = extrinsics1['fundamental_matrix']

pts0 = corners0[sample_index]
pts1 = corners1[sample_index]

ret, h0, h1 = cv2.stereoRectifyUncalibrated(pts0, pts1, fund_mat, img_size[cam1_index])  # or img_size[2]

##
cam_mat0 = intrinsics0['intrinsic_matrix']
R = np.linalg.inv(cam_mat0) @ h0 @ cam_mat0

a = cv2.getOptimalNewCameraMatrix(cam_mat0, intrinsics0['distortion_coeffs'], img_size[cam0_index], 0)
mapx, mapy = cv2.initUndistortRectifyMap(cam_mat0, intrinsics0['distortion_coeffs'], np.eye(3), a[0], img_size[cam0_index][::-1],
                                         m1type=cv2.CV_32F)

dst0 = cv2.remap(rgb_img, mapx, mapy, cv2.INTER_LINEAR)

##
cam_mat1 = intrinsics1['intrinsic_matrix']
R = np.linalg.inv(cam_mat1) @ h1  @ cam_mat1
a = cv2.getOptimalNewCameraMatrix(cam_mat1, intrinsics1['distortion_coeffs'], img_size[cam1_index], 0)
mapx, mapy = cv2.initUndistortRectifyMap(cam_mat1, intrinsics1['distortion_coeffs'], R, a[0], img_size[cam1_index][::-1],
                                         m1type=cv2.CV_32F)
dst1 = cv2.remap(ir_img, mapx, mapy, cv2.INTER_LINEAR)


plt.subplot(222), plt.imshow(ir_img)
plt.subplot(224), plt.imshow(dst1)

plt.subplot(221), plt.imshow(rgb_img)
plt.subplot(223), plt.imshow(dst0)
plt.savefig('trial.png')
plt.show()

exit()
