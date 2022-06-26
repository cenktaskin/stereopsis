from pathlib import Path
import cv2
import numpy as np
import pickle
from tqdm import tqdm

project_path = Path(__file__).joinpath("../../..").resolve()
data_path = project_path.joinpath('data')


class RawDataHandler:
    def __init__(self, data_dir, prefixes):
        self.data_dir = data_path.joinpath(data_dir)
        self.prefixes = prefixes
        self.ts_list = np.array([int(a.stem[3:]) for a in self.data_dir.glob(f'{self.prefixes[0]}*')])

    def __len__(self):
        return len(self.ts_list)

    def get_img(self, ts, cam_index, parse_st=True):
        img = cv2.imread(self.data_dir.joinpath(f"{self.prefixes[cam_index]}_{ts}.tiff").as_posix(), flags=-1)
        if cam_index < 2 and parse_st:
            img = self.parse_stereo_img(img)[cam_index]
        return img

    def get_img_size(self, cam_index):
        return self.get_img(self.ts_list[0], cam_index).shape[:2]

    def get_random_ts(self):
        return np.random.choice(self.ts_list, 1)[0]

    def get_random_img(self, cam_idx):
        return self.get_img(self.get_random_ts(), cam_idx)

    @staticmethod
    def parse_stereo_img(st_img):
        return np.split(st_img, 2, axis=1)

    def iterate_over_imgs(self):
        for ts in self.ts_list:
            yield ts, self.get_img(ts, 0, parse_st=False), self.get_img(ts, 2)


class CalibrationDataHandler(RawDataHandler):
    def __init__(self, data_id):
        super(CalibrationDataHandler, self).__init__(f"raw/calibration-{data_id}", ("st", "st", "ir"))
        self.processed_dir = data_path.joinpath("processed", f"calibration-results-{data_id}")
        if not self.processed_dir.exists():
            self.processed_dir.mkdir()
        self.frame_reviewer = FrameReviewer()

    def load_camera_info(self, cam_index, parameter_type, verbose=False):
        with open(self.processed_dir.joinpath(f"cam{cam_index}-{parameter_type}.pkl"), 'rb') as f:
            if verbose:
                print(f"Loaded {parameter_type} for camera{cam_index} from {f.name}")
            return pickle.load(f)

    def save_camera_info(self, obj, cam_index, parameter_type, verbose=False):
        with open(self.processed_dir.joinpath(f"cam{cam_index}-{parameter_type}.pkl"), 'wb') as f:
            pickle.dump(obj, f)
            if verbose:
                print(f"Dumped {parameter_type} for camera{cam_index} to {f.name}")

    def review_frame(self, fr, cam_idx=0, verbose=False):
        win_scale = 2
        if cam_idx == 2:
            win_scale = 7
        return self.frame_reviewer(fr, win_scale, verbose)


class MultipleDirDataHandler:
    def __init__(self, data_dir, prefixes):
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dir = [data_path.joinpath(d) for d in data_dir]
        self.data_handlers = [RawDataHandler(d, prefixes) for d in self.data_dir]

    def __len__(self):
        return sum([len(a) for a in self.data_handlers])

    def get_img(self, *args):
        return self.data_handlers[0].get_img(*args)

    def get_img_size(self, *args):
        return self.data_handlers[0].get_img_size(*args)

    def get_random_ts(self):
        return self.data_handlers[0].get_random_ts()

    def get_random_img(self, *args):
        return self.data_handlers[0].get_random_img(*args)

    @staticmethod
    def parse_stereo_img(self, *args):
        return self.data_handlers[0].parse_stereo_img(*args)

    def iterate_over_imgs(self):
        with tqdm(total=self.__len__()) as pbar:
            for dh in self.data_handlers:
                for ts, st, d in dh.iterate_over_imgs():
                    if not st.any():
                        pbar.update(1)
                        continue
                    yield ts, st, d
                    pbar.update(1)


class FrameReviewer:
    def __init__(self, pos_key=32, neg_key=110, quit_key=113, ):
        self.pos_key = pos_key
        self.neg_key = neg_key
        self.quit_key = quit_key
        self.valid_keys = [pos_key, neg_key, quit_key]

    def __call__(self, img, win_scale=None, verbose=False):
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        if win_scale:
            cv2.resizeWindow("img", *(np.array(img.shape)[:2] * win_scale))
        cv2.moveWindow("img", 0, 0)
        while (key_press := cv2.waitKey() & 0xFF) not in self.valid_keys:
            if verbose:
                print(f"Wrong key is pressed, probably {chr(key_press)}")
                print(f"Possible keys {[chr(i) for i in self.valid_keys]}")
                print(f"Try again...")
        cv2.destroyAllWindows()
        if key_press == self.pos_key:
            if verbose:
                print(f"Approved")
            return True
        elif key_press == self.neg_key:
            if verbose:
                print(f"Rejected")
            return False
        elif key_press == self.quit_key:
            exit()


if __name__ == "__main__":
    handler = CalibrationDataHandler("20220610")
    handler.review_frame(handler.get_random_img(0))
