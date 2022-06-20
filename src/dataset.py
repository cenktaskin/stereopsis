from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

project_path = Path(__file__).joinpath("../..").resolve()
data_path = project_path.joinpath('data')


class StereopsisDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.left_stats, self.right_stats, self.label_stats = self.get_input_stats()
        print(self.left_stats)
        self.left_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(self.left_stats[:3], self.left_stats[3:])])
        self.right_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(self.right_stats[:3], self.right_stats[3:])])
        self.target_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(self.label_stats[0], self.label_stats[1])])
        self.timestamp_list = np.array(sorted([int(x.stem[3:]) for x in self.img_dir.glob("sl*")]))

    def __len__(self):
        return len(list(self.img_dir.glob("sl*")))

    def get_input_stats(self):
        stats = np.loadtxt(self.img_dir.joinpath("stats.txt"))
        return stats[:6], stats[6:-2], stats[-2:]

    def ts_to_index(self, ts):
        return np.where(np.equal(self.timestamp_list, ts))[0][0]

    def __getitem__(self, idx):
        ts = self.timestamp_list[idx]
        image_l = Image.open(self.img_dir.joinpath(f"sl_{ts}.tiff").as_posix())
        image_r = Image.open(self.img_dir.joinpath(f"sr_{ts}.tiff").as_posix())
        label = Image.open(self.img_dir.joinpath(f"dp_{ts}.tiff").as_posix())
        if self.left_transform:
            image_l = self.left_transform(image_l)
        if self.right_transform:
            image_r = self.right_transform(image_r)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.cat([image_l, image_r]), label


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


def show_images(imgs, titles, tensor=True, row_count=1, col_count=None):
    if not col_count:
        col_count = math.ceil(len(imgs) / row_count)
    for i, img in enumerate(imgs):
        plt.subplot(row_count, col_count, i + 1)
        try:
            plt.title(f"{titles[i]} - max:{img.max():.4f}")
        except:
            pass
        if tensor:
            img = img.permute(1, 2, 0)
        plt.imshow(img, interpolation=None)
    plt.show()


if __name__ == "__main__":
    data_path = data_path.joinpath("processed/dataset-20220610-fullres/")

    dataset = StereopsisDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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
