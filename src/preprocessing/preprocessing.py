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
        self.just_resize = False

    def init_raw_size(self, sample):
        self.initial_h = sample.shape[0]
        self.initial_w = sample.shape[1]
        self.calculate_slices()
        self.define_method()

    def calculate_slices(self):
        h_crop_target = int(self.initial_w / self.target_aspect_ratio)
        reduce_h_amount = self.initial_h - h_crop_target
        if reduce_h_amount == 0:
            self.just_resize = True
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
        if not self.h_slice or not self.resize_method:
            self.init_raw_size(img)
        if not self.just_resize:
            img = self.crop_img(img)
        if img.shape != self.target_shape:
            img = self.resize_img(img)
        return img


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

    def rectify_pair(self, img0, img1, cam0_idx, cam1_idx):
        mapx0, mapy0, mapx1, mapy1 = self.calibration_data.load_camera_info(cam1_idx,
                                                                            f'rectification-map-wrt-cam{cam0_idx}')
        img0_rectified = cv2.remap(img0, mapx0, mapy0, cv2.INTER_CUBIC)
        img1_rectified = cv2.remap(img1, mapx1, mapy1, cv2.INTER_CUBIC)
        return img0_rectified, img1_rectified

    def iterate_over_imgs(self):
        with tqdm(total=self.__len__()) as pbar:
            for dh in self.data_handlers:
                for ts, st, d in dh.iterate_over_imgs():
                    if not st.any():
                        pbar.update(1)
                        continue
                    yield ts, st, d
                    pbar.update(1)

    def save_processed_imgs(self, imgs, ts):
        cv2.imwrite(self.output_path.joinpath(f"sl_{ts}.tiff").as_posix(), imgs[0])
        cv2.imwrite(self.output_path.joinpath(f"sr_{ts}.tiff").as_posix(), imgs[1])
        cv2.imwrite(self.output_path.joinpath(f"dp_{ts}.tiff").as_posix(), imgs[2])

    def crop_and_resize(self, save_result=False, verbose=False):
        target_label_res = self.target_res
        if self.output_name.split("-")[-1] == "origres":
            target_label_res = (112, 224)
        sample_resizer = ImageResizer(self.target_res, verbose=verbose)
        label_resizer = ImageResizer(target_label_res, verbose=verbose)

        stats = np.zeros(14)

        for ts, raw_st, raw_depth in self.iterate_over_imgs():
            raw_left, raw_right = np.split(raw_st, 2, axis=1)

            img_left = sample_resizer(raw_left)
            img_right = sample_resizer(raw_right)
            img_label = label_resizer(raw_depth)

            stats += np.concatenate(
                [np.array(cv2.meanStdDev(img_left)).flatten(),
                 np.array(cv2.meanStdDev(img_right)).flatten(),
                 np.array(cv2.meanStdDev(img_label)).flatten()])

            if save_result:
                self.save_processed_imgs([img_left, img_right, img_label], ts)

        stats[:-2] /= 255
        stats /= self.__len__()
        with open(self.output_path.joinpath("stats.txt"), "wb") as f:
            pickle.dump(stats, f)

    def undistort(self, save_results=False):
        maps = [self.calibration_data.load_camera_info(i, 'undistortion-map') for i in range(3)]
        sample_resizer = ImageResizer(self.target_res)
        label_resizer = ImageResizer(self.target_res)
        for ts, raw_st, raw_depth in self.iterate_over_imgs():
            raw_left, raw_right = np.split(raw_st, 2, axis=1)
            undistorted = []
            for i, img in enumerate([raw_left, raw_right, raw_depth]):
                undistorted += [cv2.remap(img, *maps[i], interpolation=cv2.INTER_CUBIC)]
            final = [sample_resizer(i[:640, :, :]) for i in undistorted[:-1]] + [label_resizer(undistorted[-1])]
            if save_results:
                self.save_processed_imgs(final, ts)

    def rectify(self, save_results=False):
        pass


if __name__ == "__main__":
    preprocessor = Preprocessor(calibration_data_id="20220610",
                                raw_datasets=["202206101932", "202206101937", "202206101612"],
                                dataset_name="20220610-undistorted")

    preprocessor.undistort(save_results=True)
