from cv2 import cv2

from dataio import save_camera_info, load_camera_info, get_random_calibration_frame
from time import time
from calibration_setup import img_size
from calibration_detection import get_img_from_dataset

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 10 ** -6)

cam0_index = 0  # fixed
cam1_index = 4

intrinsics0 = load_camera_info(cam0_index, 'intrinsics')
intrinsics1 = load_camera_info(cam1_index, 'intrinsics')
extrinsics1 = load_camera_info(cam1_index, 'extrinsics')

start_time = time()
# Only works for same size imgs
calibration_output = cv2.stereoRectify(cameraMatrix1=intrinsics0['intrinsic_matrix'],
                                       cameraMatrix2=intrinsics1['intrinsic_matrix'],
                                       distCoeffs1=intrinsics0['distortion_coeffs'],
                                       distCoeffs2=intrinsics1['distortion_coeffs'],
                                       imageSize=img_size[cam1_index],  # when imgsize=None P1 and P2 are nan
                                       R=extrinsics1['rotation_matrix'],
                                       T=extrinsics1['translation_vector'])

rectification_keys = ["R1", "R2", "P1", "P2", "Q", "roi1", "roi2"]

rectification = {x: calibration_output[i] for i, x in enumerate(rectification_keys)}

print(f"\nCamera calibrated in {time() - start_time:.2f} seconds \n"
      f"Rectification transform0: {rectification['R1']} \n\n"
      f"Rectification transform1: {rectification['R2']} \n\n"
      f"Projection matrix0:\n {rectification['P1']} \n\n"
      f"Projection matrix1:\n {rectification['P2']} \n\n"
      f"Disparity2Depth Matrix:\n {rectification['Q']} \n\n"
      f"Roi0:\n {rectification['roi1']} \n\n"
      f"Roi1:\n {rectification['roi2']}")

save_camera_info(cam1_index, 'rectification')

# how to cut with roi
#x, y, w, h = roi
#image = image[y:y + h, x:x + w]


# fixed up to here,
# I didn't understand the return values of stereoRectify so well, so I can bear on it a bit more
exit()
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics_wiki[cam0_index]['intrinsic_matrix'],
                                         distCoeffs=intrinsics_wiki[cam0_index]['distortion_coeffs'],
                                         R=rectification['R1'],
                                         newCameraMatrix=rectification['P1'],
                                         size=img_size[cam0_index][::-1],
                                         m1type=cv2.CV_32F)


sample_index = get_random_calibration_frame()

sample_image0 = get_img_from_dataset(sample_index, cam0_index)
dst = cv2.remap(sample_image0, mapx, mapy, cv2.INTER_LINEAR)
res0 = np.vstack((sample_image0, dst))
print(dst)

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics_wiki[cam1_index]['intrinsic_matrix'],
                                         distCoeffs=intrinsics_wiki[cam1_index]['distortion_coeffs'],
                                         R=rectification['R2'],
                                         newCameraMatrix=rectification['P2'],
                                         size=img_size[cam1_index][::-1],
                                         m1type=cv2.CV_32F)

sample_image1 = get_img_from_dataset(sample_index, cam1_index)
if len(sample_image1.shape) == 2:
    sample_image1 = cv2.cvtColor(sample_image1, cv2.COLOR_GRAY2BGR)
dst = cv2.remap(sample_image1, mapx, mapy, cv2.INTER_CUBIC)
res1 = np.vstack((sample_image1, dst))

result = np.hstack((res0, res1))

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", result)
cv2.waitKey()