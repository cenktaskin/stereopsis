from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from cv2 import cv2

project_path = Path.home().joinpath("git/stereopsis").resolve()
data_path = project_path.joinpath('data')


class StereopsisDataset(Dataset):
    def __init__(self, img_dir, val_split_ratio=0.01, subsample_ratio=1.0):
        self.img_dir = img_dir
        self.timestamp_list = np.array(sorted([int(x.stem[3:]) for x in self.img_dir.glob("sl*")]))
        self.len = len(list(self.img_dir.glob("sl*")))
        self.subsample_ratio = subsample_ratio
        self.train_idx, self.val_idx = self.split_validation(val_split_ratio)

    def __len__(self):
        return self.len

    def ts_to_index(self, ts):
        return np.where(np.equal(self.timestamp_list, ts))[0][0]

    def __getitem__(self, idx):
        ts = self.timestamp_list[idx]
        image_l = cv2.cvtColor(cv2.imread(self.img_dir.joinpath(f"sl_{ts}.tiff").as_posix()), cv2.COLOR_BGR2RGB)
        image_r = cv2.cvtColor(cv2.imread(self.img_dir.joinpath(f"sr_{ts}.tiff").as_posix()), cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.img_dir.joinpath(f"dp_{ts}.tiff").as_posix(), flags=cv2.IMREAD_UNCHANGED)
        return np.concatenate([image_l, image_r], axis=2).transpose((2, 0, 1)), label

    def split_validation(self, ratio):
        active_samples = int(self.len * self.subsample_ratio)
        print(f"{active_samples=}")
        np.random.shuffle(dataset_idx := list(range(self.len)))
        split_idx = int(ratio * active_samples)
        return dataset_idx[split_idx:active_samples], dataset_idx[:split_idx]

    def create_loaders(self, batch_size=16, num_workers=4):
        return DataLoader(self, batch_size=batch_size, sampler=SubsetRandomSampler(self.train_idx),
                          num_workers=num_workers), \
               DataLoader(self, batch_size=batch_size, sampler=SubsetRandomSampler(self.val_idx),
                          num_workers=num_workers)

    def assert_img_dir(self):
        if not self.img_dir.exists():
            from_home = self.img_dir.relative_to("/home/Taskin/")
            self.img_dir = Path.home().joinpath(from_home)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated


def show_images(imgs, titles=(), row_count=1, col_count=None, main_title=None, contains_bgr=False):
    if not col_count:
        col_count = math.ceil(len(imgs) / row_count)
    fig, axs = plt.subplots(row_count, col_count)
    mng = plt.get_current_fig_manager()
    for i, img in enumerate(imgs):
        img = img.squeeze()
        plt.subplot(row_count, col_count, i + 1)
        if torch.is_tensor(img):  # tensor to numpy
            img = img.numpy()
        if len(img) == 3:  # in C x H x W order
            img = img.transpose((1, 2, 0))
        if contains_bgr and img.ndim > 2:  # bgr -> rgb
            img = img[:, :, ::-1]
        try:
            plt.title(f"{titles[i]}")
        except:
            pass
        plt.imshow(img, interpolation=None)
    mng.resize(*mng.window.maxsize())
    if main_title:
        fig.suptitle(main_title)
    fig.show()
    while not plt.waitforbuttonpress():
        pass
    plt.close(fig)


if __name__ == "__main__":
    data_path = data_path.joinpath("processed/dataset-20220610-fullres/")

    ds = StereopsisDataset(data_path)
    dataloader = DataLoader(ds, batch_size=16, shuffle=True)

    train_features, train_labels = next(iter(dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    x_l, x_r = np.split(train_features[0].squeeze(), 2, axis=0)
    y = train_labels[0]

    plt.subplot(2, 2, 1)
    plt.title(f"Training sample with type {type(x_l)}, shape {x_l.shape}")
    imshow(x_l)  # back to np dimension order
    plt.subplot(2, 2, 2)
    plt.imshow(x_r.permute(1, 2, 0))
    plt.subplot(2, 2, 3)
    plt.title(f"Training sample with type {type(y)}, shape {y.shape}")
    plt.imshow(y)
    plt.show()
