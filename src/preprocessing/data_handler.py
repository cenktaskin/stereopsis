from pathlib import Path
from cv2 import cv2
import numpy as np
import yaml


class DataHandler:
    project_path = Path(__file__).joinpath("../../..").resolve()
    data_path = project_path.joinpath('data')

    def __init__(self, data_id):
        self.data_dir = self.data_path.joinpath("raw", data_id)
        self.ts_list = np.array([int(a.stem[3:]) for a in self.data_dir.glob('st*')])
        self.processed_dir = self.data_path.joinpath("processed", data_id)
        if not self.processed_dir.exists():
            self.processed_dir.mkdir()
        self.raw_ts_list = None
        self.frame_reviewer = FrameReviewer()

    def __len__(self):
        return len(self.ts_list)

    def review_frame(self, fr, cam_idx):
        win_scale = 2
        if cam_idx == 2:
            win_scale = 7
        self.frame_reviewer(fr, win_scale)

    @staticmethod
    def parse_stereo_img(st_img):
        return np.split(st_img, 2, axis=1)

    def get_img(self, ts, cam_index):
        if cam_index < 2:
            stereo_img = cv2.imread(self.data_dir.joinpath(f"st_{ts}.tiff").as_posix())
            return self.parse_stereo_img(stereo_img)[cam_index]
        elif cam_index == 2:
            return cv2.imread(self.data_dir.joinpath(f"ir_{ts}.tiff").as_posix(), flags=cv2.IMREAD_UNCHANGED)

    def load_camera_info(self, cam_index, parameter_type):
        """parameter type has to be either intrinsic or extrinsic"""
        with open(self.processed_dir.joinpath(f"cam{cam_index}-{parameter_type}.yaml"), 'r') as f:
            print(f"Loaded {parameter_type} for camera{cam_index} from {f.name}")
            return yaml.load(f, Loader=yaml.Loader)

    def save_camera_info(self, obj, cam_index, parameter_type):
        with open(self.processed_dir.joinpath(f"cam{cam_index}-{parameter_type}.yaml"), 'w') as f:
            yaml.dump(obj, f)
            print(f"Dumped {parameter_type} for camera{cam_index} to {f.name}")

    def get_random_ts(self):
        return np.random.choice(self.ts_list, 1)[0]

    def get_random_img(self, cam_idx):
        return self.get_img(self.get_random_ts(), cam_idx)


class FrameReviewer:
    def __init__(self, pos_key=32, neg_key=110, quit_key=113, verbose=False):
        self.pos_key = pos_key
        self.neg_key = neg_key
        self.quit_key = quit_key
        self.valid_keys = [pos_key, neg_key, quit_key]
        self.verbose = verbose

    def __call__(self, img, win_scale=None):
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        if win_scale:
            cv2.resizeWindow("img", np.array(img.shape) * win_scale)
        while (key_press := cv2.waitKey() & 0xFF) not in self.valid_keys:
            if self.verbose:
                print(f"Wrong key is pressed, probably {chr(key_press)}")
                print(f"Possible keys {[chr(i) for i in self.valid_keys]}")
            print(f"Try again...")
        cv2.destroyAllWindows()
        if key_press == self.pos_key:
            if self.verbose:
                print(f"Approved")
            return True
        elif key_press == self.neg_key:
            if self.neg_key:
                print(f"Rejected")
            return False
        elif key_press == self.quit_key:
            exit()


if __name__ == "__main__":
    handler = DataHandler("calibration-20220610")
    handler.review_frame(handler.get_random_img(0))

    # if False:
    #    with open(results_dir.joinpath(f"valid-frames.yaml"), 'w') as f:
    #        yaml.dump(valid_frames, f)
    #        print(f"Dumped valid frames for {data_date} to {f.name}")#
    #
    #    with open(results_dir.joinpath(f"valid-frames.yaml"), 'r') as f:
    #        print(f"Loaded valid frames for dataset {data_date} from {f.name}")
    #        a = yaml.load(f, Loader=yaml.Loader)
