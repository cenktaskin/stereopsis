from cv2 import cv2
from time import time
import numpy as np

from calibration_setup import board
from dataio import load_camera_info, save_camera_info

cam0_index = 0  # fixed
cam1_index = 2

corners0 = load_camera_info(cam0_index, 'corners')
corners1 = load_camera_info(cam1_index, 'corners')
intrinsics0 = load_camera_info(cam0_index, 'intrinsics')
intrinsics1 = load_camera_info(cam1_index, 'intrinsics')

common_frames = corners0.keys() & corners1.keys()

start_time = time()
# Calculating extrinsics
calibration_output = cv2.stereoCalibrate(objectPoints=np.tile(board, (len(common_frames), 1, 1)),
                                         imagePoints1=[corners0[frame] for frame in common_frames],
                                         imagePoints2=[corners1[frame] for frame in common_frames],
                                         cameraMatrix1=intrinsics0['intrinsic_matrix'],
                                         cameraMatrix2=intrinsics1['intrinsic_matrix'],
                                         distCoeffs1=intrinsics0['distortion_coeffs'],
                                         distCoeffs2=intrinsics1['distortion_coeffs'],
                                         imageSize=None)

extrinsics_keys = ["return_value", "_", "_", "_", "_", "rotation_matrix", "translation_vector",
                   "essential_matrix", "fundamental_matrix"]

extrinsics = {x: calibration_output[i] for i, x in enumerate(extrinsics_keys)}

print(f"\nCamera calibrated in {time() - start_time:.2f} seconds \n"
      f"Reprojection error: {extrinsics['return_value']} \n\n"
      f"R Matrix:\n {extrinsics['rotation_matrix']} \n\n"
      f"T Vects:\n {extrinsics['translation_vector']} \n\n"
      f"Essential Matrix:\n {extrinsics['essential_matrix']} \n\n"
      f"Fundamental Matrix:\n {extrinsics['fundamental_matrix']}")

save_camera_info(extrinsics, cam1_index, 'extrinsics')
