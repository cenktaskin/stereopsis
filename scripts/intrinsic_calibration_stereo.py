import numpy as np
import cv2 as cv
from pathlib import Path
import pickle

data_path = Path('/home/cenkt/projektarbeit/img_less/')
image_stems = sorted(data_path.glob('*.jpeg'))

checkerboardSize = (6, 9)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((checkerboardSize[0] * checkerboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboardSize[0], 0:checkerboardSize[1]].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane.
imgpoints_right = []

for img_stem in image_stems:
    stereo_img = cv.imread(str(img_stem))
    img_left = stereo_img[:, :stereo_img.shape[1] // 2]
    img_right = stereo_img[:, stereo_img.shape[1] // 2:]
    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
    success_left, corners_left = cv.findChessboardCorners(gray_left, checkerboardSize)
    success_right, corners_right = cv.findChessboardCorners(gray_right, checkerboardSize)
    if success_left and success_right:
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

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

intrinsics = {' left': calib_left, 'right':calib_right}

with open('intrinsic_values','wb') as handle:
    pickle.dump(intrinsics, handle)


