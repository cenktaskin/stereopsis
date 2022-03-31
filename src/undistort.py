from cv2 import cv2
import matplotlib.pyplot as plt
from random import sample
from src.data_io import data_path, load_camera_info, get_img_from_dataset, img_size


cam0_index = 0
cam1_index = 5
dataset = "20220301"

intrinsics0 = load_camera_info(cam0_index, 'intrinsics', dataset)
intrinsics1 = load_camera_info(cam1_index, 'intrinsics', dataset)
extrinsics = load_camera_info(cam1_index, 'extrinsics', dataset)
rectification = load_camera_info(cam1_index, 'rectification', dataset)

new_cam_mat0, _ = cv2.getOptimalNewCameraMatrix(intrinsics0['intrinsic_matrix'], intrinsics0['distortion_coeffs'],
                                                img_size[cam0_index], 0)
new_cam_mat1, _ = cv2.getOptimalNewCameraMatrix(intrinsics1['intrinsic_matrix'], intrinsics1['distortion_coeffs'],
                                                img_size[cam1_index], 0)

corner_frames = load_camera_info(2, 'corners', dataset).keys()
for i in range(2):
    corner_frames = corner_frames & load_camera_info(i, 'corners', dataset).keys()
frame = sample(list(corner_frames), 1)[0]
frame_path = data_path.joinpath('raw', f"calibration-{dataset}", f"st_{frame}.tiff")
imgs = [get_img_from_dataset(frame_path, i) for i in range(6)]

img0_distorted = imgs[cam0_index]
img1_distorted = imgs[cam1_index]

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics0['intrinsic_matrix'],
                                         distCoeffs=intrinsics0['distortion_coeffs'],
                                         R=rectification['R1'],
                                         newCameraMatrix=rectification['P1'],
                                         size=img_size[cam0_index][::-1],
                                         m1type=cv2.CV_32F)

img0 = cv2.remap(img0_distorted, mapx, mapy, cv2.INTER_CUBIC)

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics1['intrinsic_matrix'],
                                         distCoeffs=intrinsics1['distortion_coeffs'],
                                         R=rectification['R2'],
                                         newCameraMatrix=rectification['P1'],
                                         size=img_size[cam1_index][::-1],
                                         m1type=cv2.CV_32F)

img1 = cv2.remap(img1_distorted, mapx, mapy, cv2.INTER_CUBIC)

plt.subplot(2, 2, 1)
plt.imshow(img0_distorted[..., ::-1])
plt.subplot(2, 2, 2)
plt.imshow(img1_distorted[...])
plt.subplot(2, 2, 3)
plt.imshow(img0[..., ::-1])
plt.subplot(2, 2, 4)
plt.imshow(img1[...])
plt.show()
