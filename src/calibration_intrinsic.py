from time import time
from cv2 import cv2
import numpy as np

from calibration_setup import board, img_size
from dataio import load_camera_info, save_camera_info

camera_index = 0

dataset = '20220301'
corners_dict = load_camera_info(camera_index, 'corners', dataset)
corners = list(corners_dict.values())

print(f"Calibrating with {len(corners)} frames...")
start_time = time()
# Calculating intrinsics
calibration_output = cv2.calibrateCamera(objectPoints=np.tile(board, (len(corners), 1, 1)),
                                         imagePoints=corners, imageSize=img_size[camera_index],
                                         cameraMatrix=None, distCoeffs=None)

intrinsics_keys = ["return_value", "intrinsic_matrix", "distortion_coeffs", "rotation_vectors",
                   "translation_vectors"]
intrinsics = {x: calibration_output[i] for i, x in enumerate(intrinsics_keys)}

print(f"\nCamera calibrated in {time() - start_time:.2f} seconds \n"
      f"Reprojection error: {intrinsics['return_value']} \n\n"
      f"CameraMatrix:\n {intrinsics['intrinsic_matrix']} \n\n"
      f"Distortion Parameters:\n {intrinsics['distortion_coeffs']} \n\n"
      f"Rotation vectors (first):\n {intrinsics['rotation_vectors'][0]} \n\n"
      f"Translation vectors (first):\n {intrinsics['translation_vectors'][0]}")

save_camera_info(intrinsics, camera_index, 'intrinsics', dataset)
