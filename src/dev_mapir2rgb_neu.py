import cv2.cv2
import matplotlib.pyplot as plt
from cv2 import cv2

from dataio import load_camera_info, get_random_calibration_frame, data_path, get_img_from_dataset, upsample_ir_img, \
    parse_stereo_img
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

rgb_img = cv2.imread("st_1646059598491102197.tiff", flags=cv2.IMREAD_UNCHANGED)
left_img = parse_stereo_img(rgb_img, 0)
right_img = parse_stereo_img(rgb_img, 1)
ir_img = cv2.imread("ir_1646059598491102197.tiff", flags=cv2.IMREAD_UNCHANGED)
ir_img = upsample_ir_img(cv2.flip(ir_img, -1), resize=True)
dp_img = cv2.imread("dp_1646059598491102197.tiff", flags=cv2.IMREAD_UNCHANGED)
dp_img = upsample_ir_img(cv2.flip(dp_img, -1), resize=True)

normalized = cv2.normalize(ir_img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

print(normalized.dtype)
print(np.max(normalized))
print(np.min(normalized))
plt.figure()
plt.imshow(normalized, cmap='gray')
x = 567
y = 463
plt.plot([x], [y], marker='o', markersize=3, color="red")
plt.show()


plt.figure()
plt.imshow(dp_img, cmap='gray')
x = 567
y = 463
plt.plot([x], [y], marker='o', markersize=3, color="red")
plt.show()



plt.figure()
plt.imshow(cv2.GaussianBlur(dp_img,(5,5),0), cmap='gray')
x = 567
y = 463
plt.plot([x], [y], marker='o', markersize=3, color="red")
plt.show()

corner_ir = np.array([x, y])
corner_rgb = np.array([491.543011, 537.577419])

fig = plt.figure()
plt.imshow(left_img[:, :, ::-1])


def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    plt.plot(event.xdata, event.ydata, ',')
    fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.plot([x], [y], marker='o', markersize=3, color="red")
# plt.show()


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

# https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
left_side = np.linalg.inv(rotation_world2ir) @ np.linalg.inv(camera_matrix1) @ p_ir
right_side = np.linalg.inv(rotation_world2ir) @ translation

zconst = 2.73 * 10 ** 3 / 30
print(f"{zconst=}")
s = zconst + right_side[2] / left_side[2]
new_p = np.linalg.inv(rotation_world2ir) @ (s * np.linalg.inv(camera_matrix1) @ p_ir - translation)

img_pt = cv2.projectPoints(new_p, np.eye(3), np.zeros((3, 1)),
                           intrinsics0['intrinsic_matrix'], intrinsics0['distortion_coeffs'])

result = img_pt[0].flatten()
print(f"{result=}")
print(f"Real:{corner_rgb}")

plt.figure()
plt.imshow(left_img, cmap='jet')
x = result[0]
y = result[1]
plt.plot([x], [y], marker='o', markersize=3, color="red")
plt.show()

exit()
cv2.imshow('rgb', rgb_img)
cv2.waitKey()
cv2.imshow('ir', ir_img)
cv2.waitKey()
