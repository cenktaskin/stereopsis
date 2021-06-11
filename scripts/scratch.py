import pandas as pd
import pickle
import cv2 as cv

with open('calib_results.pkl','rb') as handle:
    results = pickle.load(handle)

mtx, dist = results['left']['mtx'], results['left']['dist']


stereo_img = cv.imread('/home/cenkt/projektarbeit/img_out/st_1623263788544123989.jpeg')
img = stereo_img[:, :stereo_img.shape[1] // 2] #left
cv.imshow('res',img)
cv.waitKey(0)

h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(newcameramtx)
print(mtx)
# undistort

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('res',dst)
cv.waitKey(0)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

