import cv2 as cv
import numpy as np
import math
import glob

checkerboardSize = (8, 5)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

obj_points = []
obj_pts_rgb = []
obj_pts_ir = []
img_points1 = []
img_points2 = []
img_shape = []
obj_p_rgb = np.zeros((checkerboardSize[0]*checkerboardSize[1], 3), np.float32)
obj_p_rgb[:, :2] = np.mgrid[0:checkerboardSize[0], 0:checkerboardSize[1]].T.reshape(-1, 2)

obj_p_ir = np.zeros((checkerboardSize[0]*checkerboardSize[1], 3), np.float32)
obj_p_ir[:, :2] = np.mgrid[0:checkerboardSize[0], 0:checkerboardSize[1]].T.reshape(-1, 2)
ct1 = 0
ct2 = 0

images1 = glob.glob('/home/rnm/Documents/filter4new1/*.jpg')
images2 = glob.glob('/home/rnm/Documents/irNew2/*.jpg')

for i in images1:
    img1 = cv.imread(i)
    img1 = cv.resize(img1, (2048, 1536))
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    success1, corners = cv.findChessboardCorners(gray1, checkerboardSize, flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

    if success1 and ct1 < 30:

        obj_pts_rgb.append(obj_p_rgb)
        corners1 = cv.cornerSubPix(gray1, corners, (16, 16), (-1, -1), criteria)
        ct1 = ct1+1
        img_points1.append(corners1)
        img = cv.drawChessboardCorners(gray1, checkerboardSize, corners1, success1)
        img_shape = gray1.shape[::-1]

    print(ct1)
    cv.imshow('img', img)
    cv.waitKey(1)
cv.destroyAllWindows()
success, cameraMatrix1, dist1, rvecs, tvecs = cv.calibrateCamera(obj_pts_rgb, img_points1, gray1.shape[::-1], cameraMatrix=None,
                                                                 distCoeffs=None, rvecs=None, tvecs=None,
                                                                 flags=cv.CALIB_RATIONAL_MODEL, criteria=criteria)
print("Camera Calibrated", success)
print("\nCameraMatrix:\n", cameraMatrix1)
print("\nDistortion Parameters:\n", dist1)
print("\nRotation vectors:\n", rvecs[0], rvecs[1], rvecs[2])
print("\nTranslation Vectors:\n", tvecs[0])


for i in images2:
    img2 = cv.imread(i)
    img2 = cv.resize(img2, (640, 576))
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    success2, corners = cv.findChessboardCorners(gray2, checkerboardSize, flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)

    if success2 and ct2 < 30:
        obj_pts_ir.append(obj_p_ir)
        corners2 = cv.cornerSubPix(gray2, corners, (16, 16), (-1, -1), criteria)
        ct2 = ct2+1
        img_points2.append(corners2)
        img = cv.drawChessboardCorners(gray2, checkerboardSize, corners2, success2)
        print(ct2)


    cv.imshow('img', img)
    cv.waitKey(1)

cv.destroyAllWindows()
success, cameraMatrix2, dist2, rvecs, tvecs = cv.calibrateCamera(obj_pts_ir, img_points2, gray1.shape[::-1], cameraMatrix=None,
                                                                 distCoeffs=None, rvecs=None, tvecs=None,
                                                                 flags=cv.CALIB_RATIONAL_MODEL, criteria=criteria)
print("Camera Calibrated", success)
print("\nCameraMatrix:\n", cameraMatrix2)
print("\nDistortion Parameters:\n", dist2)
print("\nRotation vectors:\n", rvecs[0], rvecs[1], rvecs[2])
print("\nTranslation Vectors:\n", tvecs[0])

#print("Calibrating...")

retval, cMatrix1, distcof1, cMatrix2, distcof2, R, T, E, F = cv.stereoCalibrate(obj_pts_ir, img_points1, img_points2,
                                                                                cameraMatrix1=cameraMatrix1, distCoeffs1=dist1,
                                                                                cameraMatrix2=cameraMatrix2, distCoeffs2=dist2,
                                                                                imageSize=img_shape, R=None, T=None,
                                                                                E=None, F=None, flags=cv.CALIB_RATIONAL_MODEL,
                                                                                criteria=criteria)

print("Intrinsic_mtx_1", cMatrix1)
print('dist_1', distcof1)
print('Intrinsic_mtx_2', cMatrix2)
print('dist_2', distcof2)
print('Rotation', R)
print('Translation', T)
print('E', E)
print('F', F)