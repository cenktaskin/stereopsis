from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

project_path = Path(__file__).joinpath("../..").resolve()
data_path = project_path.joinpath('data')


class StereopsisDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.timestamp_list = np.array(sorted([int(x.stem[3:]) for x in self.img_dir.glob("st*")]))

    def __len__(self):
        return len(list(self.img_dir.glob("st*")))

    def ts_to_index(self, ts):
        return np.where(np.equal(self.timestamp_list, ts))[0][0]

    def __getitem__(self, idx):
        ts = self.timestamp_list[idx]
        img_path = self.img_dir.joinpath(f"st_{ts}.tiff")
        image = cv2.cvtColor(cv2.imread(img_path.as_posix()), cv2.COLOR_BGR2RGB)
        label_path = self.img_dir.joinpath(f"dp_{ts}.tiff")
        label = cv2.imread(label_path.as_posix(), flags=cv2.IMREAD_UNCHANGED)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        l_img, r_img = torch.split(image, image.shape[2] // 2, dim=2)
        img_vol = torch.cat([l_img, r_img])
        return img_vol, label


class LabelTransformer(object):
    def __init__(self, h, w):
        self.desired_h = h
        self.desired_w = w
        self.h_slice = None
        self.w_slice = None

    @staticmethod
    def flip_label(sample):
        return cv2.flip(sample, -1)

    def crop_label(self, sample):
        if self.h_slice is None or self.w_slice is None:
            reduce_h_amount = (sample.shape[0] - self.desired_h)
            self.h_slice = slice(reduce_h_amount // 2, self.desired_h + reduce_h_amount // 2)
            self.w_slice = slice(sample.shape[1] - self.desired_w, sample.shape[1])
        return sample[self.h_slice, self.w_slice]

    def __call__(self, sample):
        sample = self.flip_label(sample)
        return self.crop_label(sample)


def parse_stereo_img(st_img, cam):
    return st_img[:, st_img.shape[1] * cam // 2:st_img.shape[1] * (cam + 1) // 2]


def np_to_tensor(sample):
    # numpy img: H x W x C -> torch img C x H x W
    sample = sample.transpose((2, 0, 1))
    return torch.from_numpy(sample)


if __name__ == "__main__":
    data_path = Path("/home/cenkt/git/stereopsis/data/raw/data-20220301/")

    label_transformer = LabelTransformer(h=120, w=214)
    dataset = StereopsisDataset(data_path, transform=transforms.Compose([np_to_tensor]),
                                target_transform=transforms.Compose([label_transformer]))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    x = train_features[0].squeeze()
    y = train_labels[0]
    plt.subplot(2, 1, 1)
    plt.title(f"Training sample with type {type(x)}, shape {x.shape}")
    plt.imshow(x.permute(1, 2, 0))  # back to np dimension order
    plt.subplot(2, 1, 2)
    plt.title(f"Training sample with type {type(y)}, shape {y.shape}")
    plt.imshow(y)
    plt.show()
