from pathlib import Path

import numpy as np
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
        return np.concatenate([image_l, image_r], axis=2).transpose((2, 0, 1)), label, idx

    def split_validation(self, ratio):
        active_samples = int(self.len * self.subsample_ratio)
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


if __name__ == "__main__":
    from preprocessing.data_io import show_images

    data_path = data_path.joinpath("processed/dataset-20220701-fullres/")

    ds = StereopsisDataset(data_path)
    dataloader, _ = ds.create_loaders()

    train_features, train_labels, idx = next(iter(dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    x_l, x_r = np.split(train_features[0].squeeze(), 2, axis=0)
    y = train_labels[0]

    show_images([x_l, x_r, y], titles=[f"Training sample with type {type(x_l)}, shape {x_l.shape}", None,
                                       f"Training sample with type {type(y)}, shape {y.shape}"], row_count=2)
