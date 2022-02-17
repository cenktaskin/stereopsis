import numpy as np
from cv2 import cv2
from pathlib import Path
import pickle
import shutil
import time

calibration_archive = Path('/home/cenkt/Documents/stereopsis/resources/calibration-images-20210609.tar.xz')
calibration_imgs_dir = Path.cwd().joinpath('../imgs', calibration_archive.name.split('.')[0])
# shutil.unpack_archive(calibration_archive,calibration_imgs_dir.parent)

# data_path = Path('/home/cenkt/projektarbeit/img_less/')
calibration_images = sorted(calibration_imgs_dir.glob('st*.jpeg'))

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chboard_size = (6, 9)
chboard = np.zeros((chboard_size[0] * chboard_size[1], 3), np.float32)
chboard[:, :2] = np.mgrid[0:chboard_size[0], 0:chboard_size[1]].T.reshape(-1, 2)

object_points, img_points_l, img_points_r = [], [], []

start_time = time.time()
i = 0
for img_file in calibration_images:
    stereo_img = cv2.imread(str(img_file))
    img_l = stereo_img[:, :stereo_img.shape[1] // 2]
    img_r = stereo_img[:, stereo_img.shape[1] // 2:]
    gray_img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    flag_l, corners_l = cv2.findChessboardCorners(gray_img_l, chboard_size)
    flag_r, corners_r = cv2.findChessboardCorners(gray_img_r, chboard_size)
    if flag_l and flag_r:
        object_points.append(chboard)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)
    else:
        print(f"Unsuccessful")
        #cv2.imshow('Stereo img',stereo_img)
        #cv2.waitKey()
        i += 1

print(f"Succcess: {i / len(calibration_images) * 100}")
print(f"Found corners in {len(calibration_images)-i} images")
print(f"Time: {time.time()-start_time} s")
exit()
print("Extracted img points")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], cameraMatrix=None,
                                                  distCoeffs=None, criteria=criteria)

print("Camera Calibrated", ret)
print("\nCameraMatrix:\n", mtx)
print("\nDistortion Parameters:\n", dist)
print("\nRotation vectors:\n", rvecs[0])
print("\nTranslation Vectors:\n", tvecs[0])

calib_left = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints_right, gray_left.shape[::-1], cameraMatrix=None,
                                                  distCoeffs=None, criteria=criteria)

print("Camera Calibrated", ret)
print("\nCameraMatrix:\n", mtx)
print("\nDistortion Parameters:\n", dist)
print("\nRotation vectors:\n", rvecs[0])
print("\nTranslation Vectors:\n", tvecs[0])

calib_right = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}

intrinsics = {' left': calib_left, 'right': calib_right}

with open('intrinsic_values', 'wb') as handle:
    pickle.dump(intrinsics, handle)
