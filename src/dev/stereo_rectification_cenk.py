import yaml
from cv2 import cv2
from time import time
from calibration_setup import *
from random import sample
from calibration_detection import get_img_from_dataset

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 10 ** -6)

extrinsics_wiki = {}
intrinsics_wiki = {}
for i in range(3):
    with open(data_path.joinpath("processed", f"cam{i}-intrinsics.yaml"), 'rb') as f:
        intrinsics_wiki[i] = yaml.load(f, Loader=yaml.Loader)
        print(f"Loaded intrinsics from {f.name}")

    if i == 0:
        continue  # since there are no extrinsics for cam0

    with open(data_path.joinpath("processed", f"cam{i}-extrinsics.yaml"), 'rb') as f:
        extrinsics_wiki[i] = yaml.load(f, Loader=yaml.Loader)
        print(f"Loaded extrinsics from {f.name}")

cam0_index = 0  # fixed
cam1_index = 2

start_time = time()


# Calculating intrinsics
def stereo_rectify(cam0_index, cam1_index):
    T = extrinsics_wiki[cam1_index]['translation_vector'].flatten()
    R = extrinsics_wiki[cam1_index]['rotation_matrix']

    # This part needs attention
    # I did it since the result was flipped and one git implementation said
    # the rot and tra is from 1->0 rather than 0->1
    R = R.T
    T = -R @ T

    e1 = T / np.linalg.norm(T)
    # print(f"{T=}")
    # print(f"{e1=}")
    e2 = np.roll(T[::-1], shift=2, axis=0) * np.array([-1, 1, 0])
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    # print(e2.shape)
    # print(f"{e2=}")
    Rrect = np.vstack((e1, e2, e3))
    # print(Rrect)
    rl = np.eye(3)
    rr = R
    Rl = Rrect @ rl
    Rr = Rrect @ rr

    # rectified camera matrix??
    # print(np.eye(3))
    # print(np.zeros)
    # print(np.vstack((T[0], np.zeros((2, 1)))))
    Pl = intrinsics_wiki[cam0_index]['intrinsic_matrix'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    Pr = intrinsics_wiki[cam1_index]['intrinsic_matrix'] @ np.hstack((np.eye(3), np.vstack((T[0], np.zeros((2, 1))))))

    return Rl, Rr, Pl, Pr


def undistort_and_show(r0, r1, p0, p1):
    calibration_imgs_dir = data_path.joinpath('raw', 'calibration-images-20210609')
    calibration_images = sorted(calibration_imgs_dir.glob('st*.jpeg'))
    sample_index = sample(calibration_images, 1)[0]

    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics_wiki[cam0_index]['intrinsic_matrix'],
                                             distCoeffs=intrinsics_wiki[cam0_index]['distortion_coeffs'],
                                             R=r0,
                                             newCameraMatrix=p0,
                                             size=img_size[cam0_index][::-1],
                                             m1type=cv2.CV_32FC1)

    sample_image0 = get_img_from_dataset(sample_index, 0)
    dst = cv2.remap(sample_image0, mapx, mapy, cv2.INTER_LINEAR)
    res0 = np.vstack((sample_image0, dst))

    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics_wiki[cam1_index]['intrinsic_matrix'],
                                             distCoeffs=intrinsics_wiki[cam1_index]['distortion_coeffs'],
                                             R=r1,
                                             newCameraMatrix=p1,
                                             size=img_size[cam1_index][::-1],
                                             m1type=cv2.CV_32F)

    sample_image1 = get_img_from_dataset(sample_index, 1)
    dst = cv2.remap(sample_image1, mapx, mapy, cv2.INTER_LINEAR)
    if cam1_index == 2:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", dst)
        cv2.resizeWindow("img", *np.array(dst.shape[:2]))
        cv2.waitKey()
    else:
        res1 = np.vstack((sample_image1, dst))

        result = np.hstack((res0, res1))

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", result)
        cv2.resizeWindow("img", *np.array(result.shape[:2]))
        cv2.waitKey()


calibration_output = cv2.stereoRectify(cameraMatrix1=intrinsics_wiki[cam0_index]['intrinsic_matrix'],
                                       cameraMatrix2=intrinsics_wiki[cam1_index]['intrinsic_matrix'],
                                       distCoeffs1=intrinsics_wiki[cam0_index]['distortion_coeffs'],
                                       distCoeffs2=intrinsics_wiki[cam1_index]['distortion_coeffs'],
                                       imageSize=img_size[0],  # when imgsize=None P1 and P2 are nan
                                       R=extrinsics_wiki[cam1_index]['rotation_matrix'],
                                       T=extrinsics_wiki[cam1_index]['translation_vector'])

rectification_keys = ["R1", "R2", "P1", "P2", "Q", "roi1", "roi2"]

rectification = {x: calibration_output[i] for i, x in enumerate(rectification_keys)}

R0, R1, P0, P1 = rectification['R1'], rectification['R2'], rectification["P1"], rectification["P2"]
print(f"{R0=}")
print(f"{R1=}")

r0, r1, p0, p1 = stereo_rectify(cam0_index, cam1_index)

print(f"{r0=}")
print(f"{r1=}")

print(R0 - r0)
print(R1 - r1)
print(P1 - p1)
print(P0 - p0)

undistort_and_show(rectification["R1"], rectification["R2"], rectification["P1"], rectification["P2"])

undistort_and_show(r0, r1, p0, p1)


# solvePnP?
# findHomography?
# getperspectivetransform??

## check whether the previous steps are valid by checking the calibraiton that is already done
# do distortion to image from cam2


# maybe try upsampling?