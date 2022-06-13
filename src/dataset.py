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
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
        self.target_transform = None
        self.timestamp_list = np.array(sorted([int(x.stem[3:]) for x in self.img_dir.glob("sl*")]))

    def __len__(self):
        return len(list(self.img_dir.glob("sl*")))

    def ts_to_index(self, ts):
        return np.where(np.equal(self.timestamp_list, ts))[0][0]

    def __getitem__(self, idx):
        ts = self.timestamp_list[idx]
        image_l = cv2.cvtColor(cv2.imread(self.img_dir.joinpath(f"sl_{ts}.tiff").as_posix()), cv2.COLOR_BGR2RGB)
        image_r = cv2.cvtColor(cv2.imread(self.img_dir.joinpath(f"sr_{ts}.tiff").as_posix()), cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.img_dir.joinpath(f"dp_{ts}.tiff").as_posix(), flags=cv2.IMREAD_UNCHANGED)
        if self.transform:
            image_l = self.transform(image_l)
            image_r = self.transform(image_r)
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
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    data_path = Path("/home/cenkt/git/stereopsis/data/processed/dataset-20220610/")

    dataset = StereopsisDataset(data_path, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
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
