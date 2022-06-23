import numpy as np
from cv2 import cv2
from data_handler import DataHandler


class Calibrator:
    st_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ir_flags = cv2.CALIB_CB_ADAPTIVE_THRESH
    board_size = (6, 9)
    square_size = 25.5  # in mm

    def __init__(self, data_id):
        self.data_handler = DataHandler(data_id)

    @staticmethod
    def create_board(n0, n1, sq_size=1.0):
        # creates a board of n0xn1
        ch_board = np.zeros((n0 * n1, 3), np.float32)
        ch_board[:, :2] = np.mgrid[0:n0, 0:n1].T.reshape(-1, 2)
        return ch_board * sq_size

    def find_corners(self, cam_idx, review=True, save_results=True):
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


if __name__ == "__main__":
    calibrator = Calibrator(data_id="calibration-20220610")
    for cam in range(3):
        calibrator.find_corners(cam_idx=2, review=True, save_results=False)
