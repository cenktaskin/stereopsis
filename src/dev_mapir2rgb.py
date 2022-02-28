from cv2 import cv2

from dataio import load_camera_info, get_random_calibration_frame, data_path, get_img_from_dataset
import numpy as np

cam0_index = 0
cam1_index = 4

intrinsics0 = load_camera_info(cam0_index, 'intrinsics')
intrinsics1 = load_camera_info(cam1_index, 'intrinsics')
extrinsics1 = load_camera_info(cam1_index, 'extrinsics')

camera_matrix0 = intrinsics0['intrinsic_matrix']
camera_matrix1 = intrinsics1['intrinsic_matrix']
rotation_world2rgb = np.eye(3)
rotation_world2ir = extrinsics1['rotation_matrix']
rotation_ir2rgb = rotation_world2rgb @ np.linalg.inv(rotation_world2ir)
translation = extrinsics1['translation_vector']

# for debug purposes get a valid frame with corners
corners0 = load_camera_info(cam0_index, 'corners')
corners1 = load_camera_info(cam1_index, 'corners')
common_frames = corners0.keys() & corners1.keys()
from random import sample

sample = sample(list(common_frames), 1)[0]
sample_index = data_path.joinpath('raw', 'calibration-images-20210609', f"st_{sample}.jpeg")

# sample_index = get_random_calibration_frame()
rgb_img = get_img_from_dataset(sample_index, cam0_index)
ir_img = get_img_from_dataset(sample_index, cam1_index)
res = np.zeros(rgb_img.shape[:-1])
# cv2.imshow('sample',sample)
# cv2.waitKey()

print(translation)
corner_ir = corners0[sample][0]
corner_rgb = corners1[sample][0]
p_ir = np.append(corner_ir, 1).reshape(3, -1)
print(f"{p_ir=}")
P_ir = np.linalg.inv(camera_matrix1) @ p_ir
print(f"{P_ir=}")
print(rotation_world2ir @ P_ir)
P_rgb = rotation_world2ir @ P_ir + translation
print(f"{P_rgb=}")
p_rgb = camera_matrix0 @ P_rgb
print(f"{p_rgb=}")
p_rgb = p_rgb / p_rgb[2]
print(f"{p_rgb=}")
print(f"{corner_rgb=}")

# https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
left_side = np.linalg.inv(rotation_world2ir) @ np.linalg.inv(camera_matrix1) @ p_ir
right_side = np.linalg.inv(rotation_world2ir) @ translation

zconst = 0.75
s = zconst + right_side[2] / left_side[2]
new_p = np.linalg.inv(rotation_world2ir) @ (s * np.linalg.inv(camera_matrix1) @ p_ir - translation)

img_pt = cv2.projectPoints(new_p, extrinsics1['rotation_matrix'], extrinsics1['translation_vector'],
                           intrinsics1['intrinsic_matrix'],intrinsics1['distortion_coeffs'])

print(img_pt[0])
print(corner_rgb)

cv2.imshow('rgb',rgb_img)
cv2.waitKey()
cv2.imshow('ir',ir_img)
cv2.waitKey()


exit()
# this is for all pts, do it later
for y in range(res.shape[0]):
    for x in range(res.shape[1]):
        p_ir = np.array([x, y, 2])
        print(f"{p_ir=}")
        P_ir = np.linalg.inv(camera_matrix1) @ p_ir
        print(f"{P_ir=}")
        print(rotation_world2ir @ P_ir)
        P_rgb = rotation_world2ir @ P_ir + translation.reshape(-1)
        print(f"{P_rgb=}")
        p_rgb = camera_matrix0 @ P_rgb
        print(f"{p_rgb=}")
        p_rgb = p_rgb / p_rgb[2]
        print(f"{p_rgb=}")
        break
    break
