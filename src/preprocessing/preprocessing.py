from cv2 import cv2
from data_io import CalibrationDataHandler, RawDataHandler
from tqdm import tqdm
import pickle
import numpy as np


class ImageResizer:
    """first crops the height to fit the aspect ratio then resizes
    ->assumes raw image height needs to be cropped, can be done robuster"""

    def __init__(self, target_size, verbose=False):
        self.initial_h = None
        self.initial_w = None
        self.target_shape = target_size
        self.target_aspect_ratio = self.target_shape[1] / self.target_shape[0]
        self.h_slice = None
        self.verbose = verbose
        self.resize_method = None

    def init_raw_size(self, sample):
        self.initial_h = sample.shape[0]
        self.initial_w = sample.shape[1]
        self.calculate_slices()
        self.define_method()

    def calculate_slices(self):
        h_crop_target = int(self.initial_w / self.target_aspect_ratio)
        reduce_h_amount = self.initial_h - h_crop_target
        self.h_slice = slice(reduce_h_amount // 2, h_crop_target + reduce_h_amount // 2)

    def define_method(self):
        method = cv2.INTER_AREA
        if self.initial_h * self.initial_w < self.target_shape[0] * self.target_shape[1]:
            method = cv2.INTER_LINEAR_EXACT
        self.resize_method = method

    def crop_img(self, img):
        res = img[self.h_slice, :]
        if self.verbose:
            print(f"Cropped {img.shape} -> {res.shape}")
        return res

    def resize_img(self, img):
        res = cv2.resize(img, self.target_shape[::-1], interpolation=self.resize_method)
        if self.verbose:
            print(f"Resized {img.shape} -> {res.shape} via method:{self.resize_method}")
        return res

    def __call__(self, img):
        if not self.h_slice:
            self.init_raw_size(img)
        cropped = self.crop_img(img)
        if cropped.shape == self.target_shape:
            return cropped
        return self.resize_img(cropped)


class Preprocessor:
    target_res = (384, 768)

    def __init__(self, calibration_data_id, raw_datasets, dataset_name):
        self.calibration_data = CalibrationDataHandler(calibration_data_id)
        self.data_path = self.calibration_data.data_path
        self.data_dirs = [self.data_path.joinpath(f"raw/rawdata-{d}") for d in raw_datasets]
        self.data_handlers = [RawDataHandler(d, ("st", "st", "dp")) for d in self.data_dirs]
        self.output_name = dataset_name
        self.output_path = self.data_path.joinpath(f"processed/dataset-{dataset_name}")
        if not self.output_path.exists():
            self.output_path.mkdir()

    def __len__(self):
        return sum([len(dh) for dh in self.data_handlers])

    def undistort(self, img, cam_idx):
        mapx, mapy = self.calibration_data.load_camera_info(cam_idx, 'undistortion-map')
        return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

    def rectify(self, img0, img1, cam0_idx, cam1_idx):
        mapx0, mapy0, mapx1, mapy1 = self.calibration_data.load_camera_info(cam1_idx,
                                                                            f'rectification-map-wrt-cam{cam0_idx}')
        img0_rectified = cv2.remap(img0, mapx0, mapy0, cv2.INTER_CUBIC)
        img1_rectified = cv2.remap(img1, mapx1, mapy1, cv2.INTER_CUBIC)
        return img0_rectified, img1_rectified

    def crop_and_resize(self, save_result=True, verbose=False):
        target_label_res = self.target_res
        if self.output_name.split("-")[-1] == "origres":
            target_label_res = (112, 224)
        sample_resizer = ImageResizer(self.target_res, verbose=verbose)
        label_resizer = ImageResizer(target_label_res, verbose=verbose)

        stats = np.zeros(14)
        with tqdm(total=self.__len__()) as pbar:
            for dh in self.data_handlers:
                for raw_st, raw_depth, ts in dh.iterate_over_imgs():
                    if not raw_st.any():
                        continue
                    raw_left, raw_right = np.split(raw_st, 2, axis=1)

                    img_left = sample_resizer(raw_left)
                    img_right = sample_resizer(raw_right)
                    img_label = label_resizer(raw_depth)

                    stats += np.concatenate(
                        [np.array(cv2.meanStdDev(img_left)).flatten(),
                         np.array(cv2.meanStdDev(img_right)).flatten(),
                         np.array(cv2.meanStdDev(img_label)).flatten()])

                    if save_result:
                        cv2.imwrite(self.output_path.joinpath(f"sl_{ts}.tiff").as_posix(), img_left)
                        cv2.imwrite(self.output_path.joinpath(f"sr_{ts}.tiff").as_posix(), img_right)
                        cv2.imwrite(self.output_path.joinpath(f"dp_{ts}.tiff").as_posix(), img_label)

                    pbar.update(1)

        stats[:-2] /= 255
        stats /= self.__len__()
        with open(self.output_path.joinpath("stats.txt"), "wb") as f:
            pickle.dump(stats, f)


if __name__ == "__main__":
    preprocessor = Preprocessor(calibration_data_id="20220610",
                                raw_datasets=["202206101932", "202206101937", "202206101612"],
                                dataset_name="20220610-undistorted")

    print(len(preprocessor))
    preprocessor.crop_and_resize(save_result=False, verbose=True)
