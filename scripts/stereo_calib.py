import cv2
import numpy as np
import cv2 as cv
from pathlib import Path
import pandas as pd
import pickle

data_path = Path('/home/cenkt/projektarbeit/img_less/')


#labels_df = pd.read_csv('review_of_calib_images1.csv').set_index('image')
#proper_images = labels_df[(labels_df['label'] == 1) | (labels_df['label'] == 3)]['label'].to_dict()

image_stems = sorted(data_path.glob('*.jpeg'))
print(image_stems)
exit()
checkerboardSize = (6, 9)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checkerboardSize[0] * checkerboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboardSize[0], 0:checkerboardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane.
imgpoints_right = []

for img_stem in image_stems:
    if np.random.rand() > 0.5:
        continue

    img_path = data_path.joinpath(img_stem + '.jpeg')
    stereo_img = cv.imread(str(img_path))

    img_left = stereo_img[:, :stereo_img.shape[1] // 2]
    img_right = stereo_img[:, stereo_img.shape[1] // 2:]

    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    success_left, corners_left = cv.findChessboardCorners(gray_left, checkerboardSize, flags=cv.CALIB_CB_FAST_CHECK)
    success_right, corners_right = cv.findChessboardCorners(gray_right, checkerboardSize, flags=cv.CALIB_CB_FAST_CHECK)

    if success_left and success_right:
        objpoints.append(objp)
        # corners_left2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
    else:
        print(img_stem, "fail")

print("Extracted img points")

retval, cMatrix1, distcof1, cMatrix2, distcof2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                                                                cameraMatrix1=None, distCoeffs1=None,
                                                                                cameraMatrix2=None, distCoeffs2=None,
                                                                                imageSize=gray_left.shape[::-1], R=None, T=None,
                                                                                E=None, F=None, flags=cv.CALIB_RATIONAL_MODEL,
                                                                                criteria=criteria)
print(retval)
print("Intrinsic_mtx_1", cMatrix1)
print('dist_1', distcof1)
print('Intrinsic_mtx_2', cMatrix2)
print('dist_2', distcof2)
print('Rotation', R)
print('Translation', T)
print('E', E)
print('F', F)

exit()
calibration_result_right = {'ret': ret,
                            'mtx': mtx,
                            'dist': dist,
                            'rvecs': rvecs,
                            'tvecs': tvecs}


calib_results = {'left': calibration_result_left, 'right': calibration_result_right}
with open('calib_results.pkl', 'wb') as handle:
    pickle.dump(calib_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
