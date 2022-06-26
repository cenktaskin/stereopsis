import numpy as np
from cv2 import cv2
from data_io import CalibrationDataHandler
import time


class Calibrator:
    st_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ir_flags = cv2.CALIB_CB_ADAPTIVE_THRESH
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.001)
    board_size = (6, 9)
    square_size = 25.5  # in mm
    intrinsics_keys = ["return_value", "intrinsic_matrix", "distortion_coeffs", "rotation_vectors",
                       "translation_vectors"]
    extrinsics_keys = ["return_value", "cam_mat1", "dist1", "cam_mat2", "dist2", "rotation_matrix",
                       "translation_vector", "essential_matrix", "fundamental_matrix"]
    rectification_keys = ["R1", "R2", "P1", "P2", "Q", "roi1", "roi2"]

    def __init__(self, data_id):
        self.data_handler = CalibrationDataHandler(data_id)
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
                    else:
                        corners = cv2.cornerSubPix(gray_img, corners, (7, 7), (-1, -1), self.criteria)
                    img_with_corners = cv2.drawChessboardCorners(img, self.board_size, corners, found)
                    if not self.data_handler.review_frame(img_with_corners, cam_idx):
                        continue
                corner_wiki[ts] = corners
            print(f"\rFrames:{i}/{len(self.data_handler)} - found:{len(corner_wiki)}", end='')

        print(f"Valid calibration frames for camera {cam_idx}: {len(corner_wiki)} / {len(self.data_handler)}")

        if save_results:
            self.data_handler.save_camera_info(corner_wiki, cam_idx, 'corners')

    def compute_intrinsics(self, cam_idx, save_results=False):
        img_size = self.data_handler.get_img_size(cam_idx)
        corners = list(self.data_handler.load_camera_info(cam_idx, 'corners').values())
        print(f"Calibrating intrinsics for cam{cam_idx} with {len(corners)} frames...")
        start_time = time.time()
        result = cv2.calibrateCamera(objectPoints=np.tile(self.board, (len(corners), 1, 1)),
                                     imagePoints=corners, cameraMatrix=None, distCoeffs=None,
                                     imageSize=img_size, flags=cv2.CALIB_RATIONAL_MODEL)
        duration = time.time() - start_time
        intrinsics = {x: result[i] for i, x in enumerate(self.intrinsics_keys)}

        print(f"Computed intrinsics in {duration:.2f} seconds \n"
              f"Reprojection error: {intrinsics['return_value']}")

        intrinsics['image_size'] = img_size

        if save_results:
            self.data_handler.save_camera_info(intrinsics, cam_idx, 'intrinsics')

    def compute_undistortion_map(self, cam_idx, save_results=False):
        intrinsics = self.data_handler.load_camera_info(cam_idx, 'intrinsics')

        new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix=intrinsics['intrinsic_matrix'],
                                                       distCoeffs=intrinsics['distortion_coeffs'],
                                                       imageSize=intrinsics['image_size'],
                                                       alpha=0)

        mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics['intrinsic_matrix'],
                                                 distCoeffs=intrinsics['distortion_coeffs'],
                                                 R=None,
                                                 newCameraMatrix=new_cam_mtx,
                                                 size=intrinsics['image_size'][::-1],
                                                 m1type=cv2.CV_32FC1)

        print(f"Computed undistortion maps for cam{cam_idx}.")

        if save_results:
            self.data_handler.save_camera_info((mapx, mapy), cam_idx, "undistortion-map")

    def compute_extrinsics(self, cam0_idx, cam1_idx, save_results=False):
        corners0 = self.data_handler.load_camera_info(cam0_idx, 'corners')
        corners1 = self.data_handler.load_camera_info(cam1_idx, 'corners')
        intrinsics0 = self.data_handler.load_camera_info(cam0_idx, 'intrinsics')
        intrinsics1 = self.data_handler.load_camera_info(cam1_idx, 'intrinsics')

        common_ts = corners0.keys() & corners1.keys()
        print(f"Calibrating extrinsics for cam{cam1_idx} wrt cam{cam0_idx} with {len(common_ts)} frames...")

        flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_SAME_FOCAL_LENGTH
        if cam1_idx == 2:
            flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_INTRINSIC

        start_time = time.time()
        result = cv2.stereoCalibrate(objectPoints=np.tile(self.board, (len(common_ts), 1, 1)),
                                     imagePoints1=[corners0[ts] for ts in common_ts],
                                     imagePoints2=[corners1[ts] for ts in common_ts],
                                     cameraMatrix1=intrinsics0['intrinsic_matrix'],
                                     cameraMatrix2=intrinsics1['intrinsic_matrix'],
                                     distCoeffs1=intrinsics0['distortion_coeffs'],
                                     distCoeffs2=intrinsics1['distortion_coeffs'],
                                     imageSize=intrinsics0['image_size'],
                                     flags=flags)

        duration = time.time() - start_time
        extrinsics = {x: result[i] for i, x in enumerate(self.extrinsics_keys)}

        print(f"Computed extrinsics in {duration:.2f} seconds \n"
              f"Reprojection error: {extrinsics['return_value']}")

        if save_results:
            self.data_handler.save_camera_info(extrinsics, cam1_idx, f'extrinsics-wrt-cam{cam0_idx}')

    def compute_rectification(self, cam0_idx, cam1_idx, save_results=False):
        intrinsics0 = self.data_handler.load_camera_info(cam0_idx, 'intrinsics')
        intrinsics1 = self.data_handler.load_camera_info(cam1_idx, 'intrinsics')
        extrinsics = self.data_handler.load_camera_info(cam1_idx, f'extrinsics-wrt-cam{cam0_idx}')

        start_time = time.time()
        result = cv2.stereoRectify(cameraMatrix1=intrinsics0['intrinsic_matrix'],
                                   cameraMatrix2=intrinsics1['intrinsic_matrix'],
                                   distCoeffs1=intrinsics0['distortion_coeffs'],
                                   distCoeffs2=intrinsics1['distortion_coeffs'],
                                   imageSize=intrinsics0['image_size'],
                                   R=extrinsics['rotation_matrix'],
                                   T=extrinsics['translation_vector'],
                                   alpha=0)

        duration = time.time() - start_time

        rectification = {x: result[i] for i, x in enumerate(self.rectification_keys)}

        print(f"Computed rectification for cam{cam0_idx} wrt cam{cam1_idx} in {duration:.2f} seconds")

        if save_results:
            self.data_handler.save_camera_info(rectification, cam1_idx, f'rectification-wrt-cam{cam0_idx}')

    def compute_rectification_maps(self, cam0_idx, cam1_idx, save_results=False):
        intrinsics0 = self.data_handler.load_camera_info(cam0_idx, 'intrinsics')
        intrinsics1 = self.data_handler.load_camera_info(cam1_idx, 'intrinsics')
        rectification = self.data_handler.load_camera_info(cam1_idx, f'rectification-wrt-cam{cam0_idx}')

        mapx0, mapy0 = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics0['intrinsic_matrix'],
                                                   distCoeffs=intrinsics0['distortion_coeffs'],
                                                   R=rectification['R1'],
                                                   newCameraMatrix=rectification['P1'],
                                                   size=intrinsics0['image_size'][::-1],
                                                   m1type=cv2.CV_32F)

        mapx1, mapy1 = cv2.initUndistortRectifyMap(cameraMatrix=intrinsics1['intrinsic_matrix'],
                                                   distCoeffs=intrinsics1['distortion_coeffs'],
                                                   R=rectification['R2'],
                                                   newCameraMatrix=rectification['P2'],
                                                   size=intrinsics1['image_size'][::-1],
                                                   m1type=cv2.CV_32F)

        print(f"Computed rectification maps for cam{cam0_idx} wrt cam{cam1_idx}")

        if save_results:
            self.data_handler.save_camera_info(((mapx0, mapy0), (mapx1, mapy1)), cam1_idx,
                                               f'rectification-map-wrt-cam{cam0_idx}')


if __name__ == "__main__":
    calibrator = Calibrator(data_id="20220610")

    corner_finding = 0
    if corner_finding:
        for cam in range(3):
            calibrator.find_corners(cam_idx=cam, review=True, save_results=False)

    intrinsic_calibration = 0
    if intrinsic_calibration:
        for cam in range(3):
            calibrator.compute_intrinsics(cam_idx=cam, save_results=False)

    undistortion_map = 0
    if undistortion_map:
        for cam in range(3):
            calibrator.compute_undistortion_map(cam_idx=cam, save_results=False)

    extrinsic_calibration = 0
    if extrinsic_calibration:
        calibrator.compute_extrinsics(cam0_idx=1, cam1_idx=2, save_results=False)

    rectification = 0
    if rectification:
        calibrator.compute_rectification(cam0_idx=1, cam1_idx=0, save_results=False)

    rectification_map = 0
    if rectification_map:
        calibrator.compute_rectification_maps(cam0_idx=1, cam1_idx=0, save_results=True)
