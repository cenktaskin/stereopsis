import numpy as np
from cv2 import cv2
from data_handler import DataHandler


class Calibrator:
    st_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ir_flags = cv2.CALIB_CB_ADAPTIVE_THRESH

    def __init__(self, data_id):
        self.board = self.create_board(6, 9, 25.5)
        self.data_handler = DataHandler(data_id)

    @staticmethod
    def create_board(n0, n1, sq_size=1.0):
        # creates a board of n0xn1
        ch_board = np.zeros((n0 * n1, 3), np.float32)
        ch_board[:, :2] = np.mgrid[0:n0, 0:n1].T.reshape(-1, 2)
        return ch_board * sq_size

    def find_corners(self, cam_idx, review=True):
        corner_wiki = {}
        i = 0
        count = 0
        for ts in self.data_handler.ts_list:
            img = self.data_handler.get_img(ts, cam_idx)
            if cam_idx < 2:  # then stereo (rgb) img
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flags = self.st_flags
            else:
                gray_img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                flags = self.ir_flags

            ret, corners = cv2.findChessboardCorners(image=gray_img, patternSize=self.board, flags=flags)
            # self.review_frame(gray_img, cam_idx)

            if ret:
                count += 1
                #if review:
                #    if not review_image(img, corners, ret, cam_index):
                #        continue
                #corner_wiki[img_file.stem[3:]] = corners

            i += 1
            # print(f"\rFrames:{i}/{len(img_paths)} - found:{len(corner_wiki)}", end='')
        print(f"{count}/{i}")
        # cv2.destroyAllWindows()
        # print(f"\r{len(corner_wiki)} valid frames for calibration")
        # return corner_wiki


if __name__ == "__main__":
    # Get the dataset
    dataset_name = '20220301'
    calibrator = Calibrator(data_id="calibration-20220610")

    calibrator.find_corners(0)
    camera_index = 5
    manual_review = True
    save_results = True

    # corners_by_frame = corner_finder(calibration_images, camera_index, review=manual_review)
    #
    # if save_results and len(corners_by_frame.keys()) > 0:
    #    save_camera_info(corners_by_frame, camera_index, 'corners', dataset_name)
