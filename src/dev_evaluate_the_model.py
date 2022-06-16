import importlib

import torch
from torch.utils.data import DataLoader

from dataset import StereopsisDataset, data_path, imshow, plot_res
import numpy as np
import matplotlib.pyplot as plt

dataset_id = "20220610"
dataset_path = data_path.joinpath(f"raw/dataset-{dataset_id}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8
data_split_ratio = 0.98
dataset = StereopsisDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model_type = "dispnet"
model_net = getattr(importlib.import_module(f"models.{model_type}"), "NNModel")
model = model_net().to(current_device)
model.load_state_dict(torch.load("dispnet_weights.pth"))
model.eval()

train_features, train_labels = next(iter(dataloader))
train_features = train_features.to(current_device)
train_labels = train_labels.to(current_device)

with torch.no_grad():
    predictions = model(train_features)

for i in range(len(train_features)):
    x_l, x_r = np.split(train_features[i].squeeze(), 2, axis=0)
    y = train_labels[i]
    y_hats = [p[i].squeeze().cpu() for p in predictions]

    plot_res([x_l.cpu(), x_r.cpu(), y.cpu()]+y_hats)
    exit()

    plt.subplot(2, 2, 1)
    imshow(x_l.cpu())  # back to np dimension order
    plt.subplot(2, 2, 2)
    imshow(x_r.cpu())
    plt.subplot(2, 2, 3)
    plt.title(f"Training sample with type {type(y)}, shape {y.shape}")
    plt.imshow(y.cpu())
    plt.subplot(2, 2, 4)
    plt.imshow(y_hat.cpu())
    plt.show()
    break
