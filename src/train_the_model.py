from datetime import datetime
import pytz
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from preprocessing.data_io import data_path
from dataset import LabelTransformer, StereopsisDataset, np_to_tensor
from loss import MaskedMSE
from src.beeline.model import BeelineModel
from torch.utils.tensorboard import SummaryWriter


def test(dataloader, model, loss_fn, device="cuda"):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            pred = model(x1, x2)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    # print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss


dataset_id = "20220301"
dataset_path = data_path.joinpath(f"raw/dataset-{dataset_id}")
train_id = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
print(f"Train id: {train_id}")
results_path = data_path.joinpath(f"processed/train-{train_id}")
if not results_path.exists():
    results_path.mkdir()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Create dataloaders
batch_size = 8
test_split_ratio = 0.9
label_transformer = LabelTransformer(h=120, w=214)
dataset = StereopsisDataset(dataset_path, transform=transforms.Compose([np_to_tensor]),
                            target_transform=transforms.Compose([label_transformer]))

train_size = int(test_split_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_idx = test_dataset.indices
#

model = BeelineModel().to(device)
loss_fn = MaskedMSE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Train the model
epochs = 2
#train_hist, test_hist = [], []
batch_count = len(train_dataloader)
print(f"Batch count: {batch_count}")
for i in range(epochs):
    train_loop = tqdm(train_dataloader, leave=False)
    train_loop.set_description(f"Epoch [{i:4d}/{epochs:4d}]")

    model.train()
    running_loss = 0
    for batch, (x1, x2, y) in enumerate(train_loop):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        prediction = model(x1, x2)
        loss = loss_fn(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        reporting_interval = 1
        if batch % reporting_interval == reporting_interval-1:
            last_loss = running_loss / reporting_interval
            #print(f"Batch [{batch+1:4d}/{batch_count:4d}] Loss: {last_loss:.4f}")
            #sample_idx = i * len(train_dataloader) + i + 1
            running_loss = 0
            train_loop.set_postfix(loss=f"{last_loss:.4f}")

    #test_loss = test(test_dataloader, model, loss_fn, device)
    #train_loop.set_postfix(loss=train_loss, test_loss=test_loss)
    #train_hist.append(train_loss)
    #test_hist.append(test_loss)
print("Done!")

#torch.save(model.state_dict(), results_path.joinpath("model.pth"))
#np.savetxt(results_path.joinpath('test_indices.txt'), test_idx)
#np.savetxt(results_path.joinpath('train_hist.txt'), train_hist)
#np.savetxt(results_path.joinpath('test_hist.txt'), test_hist)
