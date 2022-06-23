import numpy as np
from cv2 import cv2
from data_handler import DataHandler
import time


class Calibrator:
    st_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ir_flags = cv2.CALIB_CB_ADAPTIVE_THRESH
    board_size = (6, 9)
    square_size = 25.5  # in mm
    intrinsics_keys = ["return_value", "intrinsic_matrix", "distortion_coeffs", "rotation_vectors",
                       "translation_vectors"]
    extrinsics_keys = ["return_value", "_", "_", "_", "_", "rotation_matrix", "translation_vector",
                       "essential_matrix", "fundamental_matrix"]
    rectification_keys = ["R1", "R2", "P1", "P2", "Q"]

    def __init__(self, data_id):
        self.data_handler = DataHandler(data_id)
        self.board = self.create_board(*self.board_size, self.square_size)

    @staticmethod
    def create_board(n0, n1, sq_size=1.0):
        # creates a board of n0xn1
        ch_board = np.zeros((n0 * n1, 3), np.float32)
        ch_board[:, :2] = np.mgrid[0:n0, 0:n1].T.reshape(-1, 2)
        return ch_board * sq_size

    def find_corners(self, cam_idx, review=True, save_results=False):
        corner_wiki = {}
        for i, ts in enumerate(self.data_handler.ts_list):
            img = self.data_handler.get_img(ts, cam_idx)
            if cam_idx < 2:  # then stereo (rgb) img
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flags = self.st_flags
            else:
                gray_img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                flags = self.ir_flags

            found, corners = cv2.findChessboardCorners(image=gray_img, patternSize=self.board_size, flags=flags)
            if found:
                if review:
                    if cam_idx == 2:  # show the normalized img for ir channel
                        img = gray_img
                    img_with_corners = cv2.drawChessboardCorners(img, self.board_size, corners, found)
                    if not self.data_handler.review_frame(img_with_corners, cam_idx):
                        continue
                corner_wiki[ts] = corners
            print(f"\rFrames:{i}/{len(self.data_handler)} - found:{len(corner_wiki)}", end='')

        print(f"Valid calibration frames for camera {cam_idx}: {len(corner_wiki)} / {len(self.data_handler)}")
        if save_results:
            self.data_handler.save_camera_info(corner_wiki, cam_idx, 'corners')

    def compute_intrinsics(self, cam_idx, save_results=False):
        corners = list(self.data_handler.load_camera_info(cam_idx, 'corners').values())
        print(f"Calibrating with {len(corners)} frames...")
        start_time = time.time()
        result = cv2.calibrateCamera(objectPoints=np.tile(self.board, (len(corners), 1, 1)),
                                     imagePoints=corners, cameraMatrix=None, distCoeffs=None,
                                     imageSize=self.data_handler.get_random_img(cam_idx).shape[:2])
        duration = time.time() - start_time
        intrinsics = {x: result[i] for i, x in enumerate(self.intrinsics_keys)}

        print(f"\nCamera calibrated in {duration:.2f} seconds \n"
              f"Reprojection error: {intrinsics['return_value']} \n\n"
              f"CameraMatrix:\n {intrinsics['intrinsic_matrix']} \n\n"
              f"Distortion Parameters:\n {intrinsics['distortion_coeffs']} \n\n"
              f"Rotation vectors (first):\n {intrinsics['rotation_vectors'][0]} \n\n"
              f"Translation vectors (first):\n {intrinsics['translation_vectors'][0]}")

        if save_results:
            self.data_handler.save_camera_info(intrinsics, cam_idx, 'intrinsics')

    def compute_extrinsics(self, cam0_idx, cam1_idx, save_results=False):
        corners0 = self.data_handler.load_camera_info(cam0_idx, 'corners')
        corners1 = self.data_handler.load_camera_info(cam1_idx, 'corners')
        intrinsics0 = self.data_handler.load_camera_info(cam0_idx, 'intrinsics')
        intrinsics1 = self.data_handler.load_camera_info(cam1_idx, 'intrinsics')

        common_ts = corners0.keys() & corners1.keys()
        print(f"Calibrating with {len(common_ts)} frames...")

        start_time = time.time()
        result = cv2.stereoCalibrate(objectPoints=np.tile(self.board, (len(common_ts), 1, 1)),
                                     imagePoints1=[corners0[ts] for ts in common_ts],
                                     imagePoints2=[corners1[ts] for ts in common_ts],
                                     cameraMatrix1=intrinsics0['intrinsic_matrix'],
                                     cameraMatrix2=intrinsics1['intrinsic_matrix'],
                                     distCoeffs1=intrinsics0['distortion_coeffs'],
                                     distCoeffs2=intrinsics1['distortion_coeffs'],
                                     imageSize=None)
        duration = time.time() - start_time
        extrinsics = {x: result[i] for i, x in enumerate(self.extrinsics_keys)}
        print(f"\nCamera calibrated in {duration:.2f} seconds \n"
              f"Reprojection error: {extrinsics['return_value']} \n\n"
              f"R Matrix:\n {extrinsics['rotation_matrix']} \n\n"
              f"T Vects:\n {extrinsics['translation_vector']} \n\n"
              f"Essential Matrix:\n {extrinsics['essential_matrix']} \n\n"
              f"Fundamental Matrix:\n {extrinsics['fundamental_matrix']}")

        if save_results:
            self.data_handler.save_camera_info(extrinsics, cam1_idx, 'extrinsics')

    def compute_rectification(self, cam0_idx, cam1_idx, save_results=False):
        intrinsics0 = self.data_handler.load_camera_info(cam0_idx, 'intrinsics')
        intrinsics1 = self.data_handler.load_camera_info(cam1_idx, 'intrinsics')
        extrinsics = self.data_handler.load_camera_info(cam1_idx, 'extrinsics')

        start_time = time.time()
        result = cv2.stereoRectify(cameraMatrix1=intrinsics0['intrinsic_matrix'],
                                   cameraMatrix2=intrinsics1['intrinsic_matrix'],
                                   distCoeffs1=intrinsics0['distortion_coeffs'],
                                   distCoeffs2=intrinsics1['distortion_coeffs'],
                                   imageSize=self.data_handler.get_random_img(cam1_idx).shape[:2],
                                   R=extrinsics['rotation_matrix'],
                                   T=extrinsics['translation_vector'],
                                   alpha=0)

        duration = time.time() - start_time

        rectification = {x: result[i] for i, x in enumerate(self.rectification_keys)}

        print(f"\nCamera calibrated in {duration:.2f} seconds \n"
              f"Rectification transform0: {rectification['R1']} \n\n"
              f"Rectification transform1: {rectification['R2']} \n\n"
              f"Projection matrix0:\n {rectification['P1']} \n\n"
              f"Projection matrix1:\n {rectification['P2']} \n\n"
              f"Disparity2Depth Matrix:\n {rectification['Q']}")

        if save_results:
            self.data_handler.save_camera_info(rectification, cam1_idx, 'rectification')


if __name__ == "__main__":
    calibrator = Calibrator(data_id="calibration-20220610")

    find_corner = False
    if find_corner:
        for cam in range(3):
            calibrator.find_corners(cam_idx=cam, review=True, save_results=False)

    intrinsic_calibration = False
    if intrinsic_calibration:
        for cam in range(3):
            calibrator.compute_intrinsics(cam_idx=cam, save_results=False)

    extrinsic_calibration = False
    if extrinsic_calibration:
        calibrator.compute_extrinsics(cam0_idx=1, cam1_idx=0, save_results=False)

    rectification = True
    if rectification:
        calibrator.compute_rectification(cam0_idx=1, cam1_idx=0, save_results=False)
