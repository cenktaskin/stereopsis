from cv2 import cv2

from data_io import save_camera_info, load_camera_info, img_size
from time import time
from calibration_detection import get_img_from_dataset

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 10 ** -6)
dataset = "20220301"
cam0_index = 0  # fixed
cam1_index = 1

intrinsics0 = load_camera_info(0, 'intrinsics', dataset)
intrinsics1 = load_camera_info(1, 'intrinsics', dataset)
extrinsics = load_camera_info(1, 'extrinsics', dataset)

start_time = time()
# Only works for same size imgs
calibration_output = cv2.stereoRectify(cameraMatrix1=intrinsics0['intrinsic_matrix'],
                                       cameraMatrix2=intrinsics1['intrinsic_matrix'],
                                       distCoeffs1=intrinsics0['distortion_coeffs'],
                                       distCoeffs2=intrinsics1['distortion_coeffs'],
                                       imageSize=img_size[cam1_index],  # when imgsize=None P1 and P2 are nan
                                       R=extrinsics['rotation_matrix'],
                                       T=extrinsics['translation_vector'],
                                       alpha=0)

rectification_keys = ["R1", "R2", "P1", "P2", "Q"]

rectification = {x: calibration_output[i] for i, x in enumerate(rectification_keys)}

print(f"\nCamera calibrated in {time() - start_time:.2f} seconds \n"
      f"Rectification transform0: {rectification['R1']} \n\n"
      f"Rectification transform1: {rectification['R2']} \n\n"
      f"Projection matrix0:\n {rectification['P1']} \n\n"
      f"Projection matrix1:\n {rectification['P2']} \n\n"
      f"Disparity2Depth Matrix:\n {rectification['Q']}")

save_camera_info(rectification, cam1_index, 'rectification', dataset)
