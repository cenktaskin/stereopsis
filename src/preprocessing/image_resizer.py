import numpy as np
from src.dataset import data_path
from tqdm import tqdm
from cv2 import cv2
import os


class ImageResizer(object):
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


if __name__ == "__main__":
    dataset_type = "fullres"  # fullres or origres
    # after making sure it works, you can turns this into glob
    acquired_datasets = ["202206101932", "202206101937", "202206101612"]
    data_dirs = [data_path.joinpath(f"raw/rawdata-{d}") for d in acquired_datasets]
    data_dirs = [data_dirs[1]]
    total_imgs = sum([sum([len(files) for _, _, files in os.walk(i)]) for i in data_dirs]) // 2

    target_sample_res = (384, 768)
    target_label_res = target_sample_res
    if dataset_type == "origres":
        target_label_res = (112, 224)
    sample_resizer = ImageResizer(target_sample_res)
    label_resizer = ImageResizer(target_label_res)

    clean_dataset_path = data_path.joinpath(f"processed/dataset-20220610-{dataset_type}/")
    if not clean_dataset_path.exists():
        clean_dataset_path.mkdir()

    stats = np.zeros(14)
    with tqdm(total=total_imgs) as pbar:
        for data_dir in data_dirs:
            for img_path in data_dir.glob("st_*.tiff"):
                ts = img_path.stem[3:]
                raw_image = cv2.imread(img_path.as_posix())
                if not raw_image.any():
                    # in the last dataset there were mjpeg errors causing some frames to be emtpy
                    continue

                raw_left, raw_right = np.split(raw_image, 2, axis=1)
                label_path = img_path.with_stem(f"dp_{ts}")
                raw_label = cv2.imread(label_path.as_posix(), flags=cv2.IMREAD_UNCHANGED)

                img_left = sample_resizer(raw_left)
                img_right = sample_resizer(raw_right)
                img_label = label_resizer(raw_label)

                stats += np.concatenate(
                    [np.array(cv2.meanStdDev(img_left)).flatten(),
                     np.array(cv2.meanStdDev(img_right)).flatten(),
                     np.array(cv2.meanStdDev(img_label)).flatten()]) / total_imgs
                # cv2.imwrite(clean_dataset_path.joinpath(f"sl_{ts}.tiff").as_posix(), img_left)
                # cv2.imwrite(clean_dataset_path.joinpath(f"sr_{ts}.tiff").as_posix(), img_right)
                # cv2.imwrite(clean_dataset_path.joinpath(f"dp_{ts}.tiff").as_posix(), img_label)
                pbar.update(1)

    stats[:-2] /= 255
    np.savetxt(clean_dataset_path.joinpath("stats.txt"), stats)
