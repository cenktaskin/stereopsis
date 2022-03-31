from cv2 import cv2
import matplotlib.pyplot as plt
from src.data_io import load_camera_info, get_img_from_dataset, data_path
import numpy as np
from random import sample


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        print(tuple(pt1[0]))
        img1 = cv2.circle(img1, tuple(np.round(pt1[0]).astype(int)), 1, color, -1)
        img2 = cv2.circle(img2, tuple(np.round(pt2[0]).astype(int)), 5, color, -1)
    return img1, img2


cam0_index = 0
cam1_index = 2

corners0 = load_camera_info(cam0_index, 'corners')
corners1 = load_camera_info(cam1_index, 'corners')
intrinsics0 = load_camera_info(cam0_index, 'intrinsics')
intrinsics1 = load_camera_info(cam1_index, 'intrinsics')
extrinsics1 = load_camera_info(cam1_index, 'extrinsics')

common_frames = corners0.keys() & corners1.keys()
sample_index = sample(list(common_frames), 1)[0]
sample_path = data_path.joinpath("raw", "calibration-images-20210609", f"st_{sample_index}.jpeg")
rgb_img = get_img_from_dataset(sample_path, cam0_index)
ir_img = get_img_from_dataset(sample_path, cam1_index)

fund_mat = extrinsics1['fundamental_matrix']

pts0 = corners0[sample_index]
pts1 = corners1[sample_index]

rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, fund_mat)
print(f"{lines1=}")
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(rgb_img, ir_img, lines1, list(pts0), list(pts1))
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts0.reshape(-1, 1, 2), 1, fund_mat)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(ir_img, rgb_img, lines2, list(pts1), list(pts0))
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.savefig('trial.png')
plt.show()
