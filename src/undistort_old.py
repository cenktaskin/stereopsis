import cv2 as cv
from pathlib import Path
import yaml
import numpy as np

data_path = Path('/home/cenkt/projektarbeit/calibrationdata/')

with open(data_path.joinpath('left.yaml'), 'r') as stream:
    left_yaml = yaml.safe_load(stream)
with open(data_path.joinpath('right.yaml'), 'r') as stream:
    right_yaml = yaml.safe_load(stream)


mtx_left = np.array(left_yaml['camera_matrix']['data']).reshape(3,3)
dist_left = np.array(left_yaml['distortion_coefficients']['data'])

mtx_right = np.array(right_yaml['camera_matrix']['data']).reshape(3,3)
dist_right = np.array(right_yaml['distortion_coefficients']['data'])

img_left = cv.imread(str(data_path.joinpath('left-0069.png')))
img_right = cv.imread(str(data_path.joinpath('right-0069.png')))

h,  w = img_left.shape[:2]
newcameramtx_left, _ = cv.getOptimalNewCameraMatrix(mtx_left, dist_left, (w,h), 1, (w,h))
newcameramtx_right, _ = cv.getOptimalNewCameraMatrix(mtx_right, dist_right, (w,h), 1, (w,h))
# undistort
dst_left = cv.undistort(img_left, mtx_left, dist_left, None, newcameramtx_left)
dst_right = cv.undistort(img_right, mtx_right, dist_right, None, newcameramtx_right)

top = np.hstack((img_left,img_right))
bottom = np.hstack((dst_left,dst_right))
numpy_vertical = np.vstack((top,bottom))
cv.imshow('calibration_result', numpy_vertical)
cv.imwrite('result.png',numpy_vertical)
cv.waitKey(0)