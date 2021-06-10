import numpy as np
import cv2 as cv
from pathlib import Path

data_path = Path('/home/cenkt/projektarbeit/calib_images/')
image_paths = []

for img_path in data_path.glob("st_*.jpeg"):
    image_paths.append(img_path)

image_paths = sorted(image_paths)

checkerboardSize = (6,9)

# noinspection PyUnresolvedReferences
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # ??

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checkerboardSize[0]*checkerboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboardSize[0], 0:checkerboardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

target_cam = 'left'

for file_name in image_paths:
    stereo_img = cv.imread(str(file_name))

    img = stereo_img[:,:stereo_img.shape[1]//2]
    if target_cam == 'right':
        img = stereo_img[:,stereo_img.shape[1]//2:]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, checkerboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, checkerboardSize, corners2, ret)
        cv.imshow(file_name.stem, img)
        cv.waitKey(100)
    else:
        print(file_name.stem ,"FAIL")

    cv.destroyAllWindows()