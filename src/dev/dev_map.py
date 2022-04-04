import matplotlib.pyplot as plt
from cv2 import cv2

from src.data_io import load_camera_info, data_path, get_img_from_dataset, img_size
import numpy as np
from random import sample


def back_project(px, z, cam_mat, dst_coeffs, rot_mat, trans, opt_cam_mat):
    p_ir = cv2.undistortPoints(np.array(px, dtype=np.float64), cam_mat, dst_coeffs, R=rot_mat, P=opt_cam_mat)
    p_ir = np.append(p_ir, 1).reshape(3, -1)

    left_side = np.linalg.inv(rot_mat) @ np.linalg.inv(opt_cam_mat) @ p_ir
    right_side = np.linalg.inv(rot_mat) @ trans

    depth_at_point = z * 10 ** 3
    s = depth_at_point + right_side[2] / left_side[2]
    return np.linalg.inv(rot_wto2) @ (s * np.linalg.inv(new_cmat) @ p_ir - tra)


cam0_index = 0
cam1_index = 2
dataset = "20220301"

intrinsics0 = load_camera_info(cam0_index, 'intrinsics', dataset)
intrinsics1 = load_camera_info(cam1_index, 'intrinsics', dataset)
extrinsics1 = load_camera_info(cam1_index, 'extrinsics', dataset)

corner_frames = load_camera_info(2, 'corners', dataset).keys()
for i in range(2):
    corner_frames = corner_frames & load_camera_info(i, 'corners', dataset).keys()

frame = sample(list(corner_frames), 1)[0]

frame_path = data_path.joinpath('raw', f"calibration-{dataset}", f"st_{frame}.tiff")

imgs = [get_img_from_dataset(frame_path, i) for i in range(4)]

camera_matrix0 = intrinsics0['intrinsic_matrix']
camera_matrix1 = intrinsics1['intrinsic_matrix']
rot_wto0 = np.eye(3)
rot_wto2 = extrinsics1['rotation_matrix']
rot_2to0 = rot_wto0 @ np.linalg.inv(rot_wto2)
tra = extrinsics1['translation_vector']

new_cmat, _ = cv2.getOptimalNewCameraMatrix(camera_matrix1, intrinsics1['distortion_coeffs'], img_size[2], 1)

depth_img = imgs[3]

# you can test the corners and iterate onyl in btw
depth_layer = np.zeros_like(imgs[0][:, :, 0], dtype=float)

display = False
for r in range(0, depth_img.shape[0]):
    for c in range(depth_img.shape[1]):
        if depth_img[r][c] <= 0:
            continue

        new_p = back_project([c, r], depth_img[r][c], camera_matrix1, intrinsics1['distortion_coeffs'], rot_wto2,
                             extrinsics1['translation_vector'], new_cmat)

        img_pt = cv2.projectPoints(new_p, np.eye(3), np.zeros((3, 1)),
                                   intrinsics0['intrinsic_matrix'], intrinsics0['distortion_coeffs'])

        result = img_pt[0].flatten()
        end_px = np.round(result).astype(int)

        if np.any(end_px < 0):
            continue

        if np.any((np.array(img_size[0])[::-1] - end_px) <= 0):
            continue

        # interpolating will be better
        depth_layer[end_px[1], end_px[0]] = depth_img[r, c]

        #print(f"{result=}")
        if display:
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(imgs[0][:, :, ::-1])
            plt.plot(*result, marker='x', markersize=3, color="blue")
            plt.subplot(1, 2, 2)
            plt.imshow(imgs[3])
            plt.plot(c, r, marker='o', markersize=3, color="red")
            figManager = plt.get_current_fig_manager()
            figManager.resize(*figManager.window.maxsize())
            plt.show()
hist = False
if hist:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(depth_layer)
    plt.subplot(1, 2, 2)
    plt.hist(depth_img)
    plt.show()


interp_depth = depth_layer.copy()
interp_depth[interp_depth == 0] = np.nan
from scipy import interpolate

x = np.arange(0, interp_depth.shape[1])
y = np.arange(0, interp_depth.shape[0])

array = np.ma.masked_invalid(interp_depth)
xx, yy = np.meshgrid(x, y)
# get only the valid values
x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]

#GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
#                           (xx, yy),
#                           method='cubic')

fig, axes = plt.subplots(3, 2)
axes[0][0].set_title("Depth img")
axes[0][0].imshow(depth_img, cmap='gray')
axes[0][1].set_title("Result of mapping")
axes[0, 1].imshow(depth_layer)
axes[1][0].set_title("Interpolated1")
axes[1][0].imshow(imgs[0][:,:,::-1])


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


res3 = interpolate_missing_pixels(depth_layer, depth_layer == 0.0, 'cubic')

axes[1][1].set_title("Interpolated")
axes[1,1].imshow(res3,cmap='gray')

axes[2,0].set_title("Hist labels")
axes[2,0].hist(depth_img)
axes[2,1].set_title("Hist interpolated")
axes[2,1].hist(res3)

plt.show()

exit()
# normalized_depth = cv2.normalize(depth_layer, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
plt.imshow(depth_layer, interpolation='lanczos', cmap='gray')
plt.show()
exit()

normalized_depth = cv2.normalize(depth_layer, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(imgs[0][:, :, ::-1])
plt.subplot(1, 2, 2)
plt.imshow(normalized_depth)
figManager = plt.get_current_fig_manager()
figManager.resize(*figManager.window.maxsize())
plt.show()
