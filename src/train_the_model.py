from datetime import datetime
import pytz
import numpy as np

import torch
from preprocessing.data_io import data_path
from loss import MaskedMSE
from models import BeelineModel
from train_wrappers import create_dataloaders, train_the_model


dataset_id = "20220301"
dataset_path = data_path.joinpath(f"raw/dataset-{dataset_id}")
train_id = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
print(f"Train id: {train_id}")
results_path = data_path.joinpath(f"processed/train-{train_id}")
if not results_path.exists():
    results_path.mkdir()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_dataloader, test_dataloader, test_idx = create_dataloaders(dataset_path, batch_size=32, test_split_ratio=0.9)

model = BeelineModel().to(device)

loss_fn = MaskedMSE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

epochs = 200
model, train_hist, test_hist = train_the_model(model, epochs, train_dataloader, test_dataloader, loss_fn, optimizer, device)

torch.save(model.state_dict(), results_path.joinpath("model.pth"))
np.savetxt(results_path.joinpath('test_indices.txt'), test_idx)
np.savetxt(results_path.joinpath('train_hist.txt'), train_hist)
np.savetxt(results_path.joinpath('test_hist.txt'), test_hist)
