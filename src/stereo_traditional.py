from data_io import get_img_from_dataset, load_camera_info, data_path, img_size
import numpy as np
import yaml
from cv2 import cv2
import matplotlib.pyplot as plt
from random import sample


dataset = "20220301"
dataset_path = data_path.joinpath("raw", f"data-{dataset}")
intrinsics0 = load_camera_info(0, 'intrinsics', dataset)
intrinsics1 = load_camera_info(1, 'intrinsics', dataset)
extrinsics = load_camera_info(1, 'extrinsics', dataset)
rectification = load_camera_info(1, 'rectification', dataset)

new_cam_mat0, _ = cv2.getOptimalNewCameraMatrix(intrinsics0['intrinsic_matrix'], intrinsics0['distortion_coeffs'],
                                                img_size[0], 0)
new_cam_mat1, _ = cv2.getOptimalNewCameraMatrix(intrinsics1['intrinsic_matrix'], intrinsics1['distortion_coeffs'],
                                                img_size[0], 0)

with open(data_path.joinpath("processed", dataset, f"valid-frames.yaml"), 'r') as f:
    print(f"Loaded valid frames for dataset {dataset} from {f.name}")
    valid_frames = yaml.load(f, Loader=yaml.Loader)

# for fr in valid_frames:
fr = sample(valid_frames, 1)[0]
#fr = valid_frames[347]
img_path = dataset_path.joinpath(f"st_{fr}.tiff")
img0_distorted = get_img_from_dataset(img_path, 0)
img1_distorted = get_img_from_dataset(img_path, 1)

img_depth = get_img_from_dataset(img_path, 3)

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics0['intrinsic_matrix'],
                                         distCoeffs=intrinsics0['distortion_coeffs'],
                                         R=rectification['R1'],
                                         newCameraMatrix=rectification['P1'],
                                         size=img_size[0][::-1],
                                         m1type=cv2.CV_32F)

img0 = cv2.remap(img0_distorted, mapx, mapy, cv2.INTER_CUBIC)

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics1['intrinsic_matrix'],
                                         distCoeffs=intrinsics1['distortion_coeffs'],
                                         R=rectification['R2'],
                                         newCameraMatrix=rectification['P1'],
                                         size=img_size[1][::-1],
                                         m1type=cv2.CV_32F)

img1 = cv2.remap(img0_distorted, mapx, mapy, cv2.INTER_CUBIC)

display = True
if display:
    plt.subplot(2, 2, 1)
    plt.imshow(img0_distorted[..., ::-1])
    plt.subplot(2, 2, 2)
    plt.imshow(img1_distorted[..., ::-1])
    plt.subplot(2, 2, 3)
    plt.imshow(img0[..., ::-1])
    plt.subplot(2, 2, 4)
    plt.imshow(img1[..., ::-1])
    plt.show()

# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 16
num_disp = 256 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               #blockSize=11,
                               #P1=8 * 3 * window_size ** 2,
                               #P2=32 * 3 * window_size ** 2,
                               #disp12MaxDiff=15,
                               #uniquenessRatio=15,
                               #speckleWindowSize=100,
                               #preFilterCap=63,
                               #speckleRange=2,
                               mode=cv2.STEREO_SGBM_MODE_HH4
                               )



img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

disparity = stereo.compute(img0_gray, img1_gray)

depth = cv2.reprojectImageTo3D(disparity, rectification['Q'], handleMissingValues=False)/100

plt.subplot(2, 2, 1)
plt.imshow(img0[..., ::-1])
plt.subplot(2, 2, 2)
plt.imshow(img_depth, cmap='viridis')
plt.subplot(2, 2, 3)
plt.imshow(depth[:, :, 2], cmap='viridis')
figManager = plt.get_current_fig_manager()
figManager.resize(*figManager.window.maxsize())
plt.show()
