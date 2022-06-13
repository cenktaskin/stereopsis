import torch
from torchvision import transforms
from models.beeline import NNModel
from dataset import data_path, LabelTransformer, DataLoader, StereopsisDataset, np_to_tensor
import numpy as np
import matplotlib.pyplot as plt

log_dir = list(data_path.glob('runs/beeline4-*/'))[0]
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NNModel('test', [64, 128, 256, 512])
model.load_state_dict(torch.load(log_dir.joinpath('model.pth'), map_location=current_device))

label_transformer = LabelTransformer(h=120, w=214)
dataset = StereopsisDataset(data_path.joinpath("raw/dataset-20220301/"), transform=transforms.Compose([np_to_tensor]),
                            target_transform=transforms.Compose([label_transformer]))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
train_features, train_labels = next(iter(dataloader))

model.eval()
with torch.no_grad():
    pred = model(train_features)

for i in range(len(train_features)):
    x_l, x_r = np.split(train_features[i].squeeze(), 2, axis=0)
    y = train_labels[i]
    y_hat = pred[i].squeeze()

    plt.subplot(2, 2, 1)
    plt.title(f"Training sample with type {type(x_l)}, shape {x_l.shape}")
    plt.imshow(x_l.permute(1, 2, 0))  # back to np dimension order
    plt.subplot(2, 2, 2)
    plt.imshow(x_r.permute(1, 2, 0))
    plt.subplot(2, 2, 3)
    plt.title(f"Training sample with type {type(y)}, shape {y.shape}")
    plt.imshow(y)
    plt.subplot(2, 2, 4)
    plt.imshow(y_hat)
    plt.show()
