from time import time
from cv2 import cv2
import numpy as np

from calibration_setup import board, img_size
from dataio import load_camera_info, save_camera_info

camera_index = 4

corners_dict = load_camera_info(camera_index, 'corners')
corners = list(corners_dict.values())

# downsample the corners if needed
# frames = sample(list(all_corners_dict.keys()), max_frame_count)
# corners = [all_corners_dict[k] for k in frames]

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

save_camera_info(camera_index, 'intrinsics')
