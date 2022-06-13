import numpy as np
from dataset import data_path
from tqdm import tqdm
from cv2 import cv2
import os


class LabelTransformer(object):
    def __init__(self, h, w):
        self.desired_h = h
        self.desired_w = w
        self.h_slice = None
        self.w_slice = None

    def crop_label(self, sample):
        if self.h_slice is None:
            reduce_h_amount = (sample.shape[0] - self.desired_h)
            self.h_slice = slice(reduce_h_amount // 2, self.desired_h + reduce_h_amount // 2)
            self.w_slice = slice(sample.shape[1] - self.desired_w, sample.shape[1])
        return sample[self.h_slice, self.w_slice]

    def __call__(self, sample):
        return self.crop_label(sample)


# after making sure it works, you can turns this into glob
acquired_datasets = ["202206101932", "202206101937", "202206101612"]
data_dirs = [data_path.joinpath(f"raw/dataset-{d}") for d in acquired_datasets]

total_imgs = sum([sum([len(files) for _, _, files in os.walk(i)]) for i in data_dirs]) // 2

label_transformer = LabelTransformer(120, 214)

clean_dataset_path = data_path.joinpath(f"processed/dataset-20220610/")
if not clean_dataset_path.exists():
    clean_dataset_path.mkdir()

with tqdm(total=total_imgs) as pbar:
    for data_dir in data_dirs:
        for img_path in data_dir.glob("st_*.tiff"):
            ts = img_path.stem[3:]
            image = cv2.imread(img_path.as_posix())
            if not image.any():
                continue
            label_path = img_path.with_stem(f"dp_{ts}")
            label = cv2.imread(label_path.as_posix(), flags=cv2.IMREAD_UNCHANGED)

            l_img, r_img = np.split(image, 2, axis=1)
            label = label_transformer(label)

            cv2.imwrite(clean_dataset_path.joinpath(f"sl_{ts}.tiff").as_posix(), l_img)
            cv2.imwrite(clean_dataset_path.joinpath(f"sr_{ts}.tiff").as_posix(), r_img)
            cv2.imwrite(clean_dataset_path.joinpath(f"dp_{ts}.tiff").as_posix(), label)
            pbar.update(1)
