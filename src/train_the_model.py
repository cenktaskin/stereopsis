from datetime import datetime
import pytz
from tqdm import tqdm
import importlib

import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import StereopsisDataset, data_path
from dispnet_loss import MaskedMSE

dataset_id = "20220610"
dataset_path = data_path.joinpath(f"raw/dataset-{dataset_id}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8
data_split_ratio = 0.98
dataset = StereopsisDataset(dataset_path)

train_size = int(data_split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model_type = "dispnet"
model_net = getattr(importlib.import_module(f"models.{model_type}"), "NNModel")
model = model_net().to(current_device)

loss_fn = MaskedMSE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
# scheduler should rather start decaying after 400k according to paper but this is more useful
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2 * 10 ** 5, gamma=0.5)

timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
results_path = data_path.joinpath(f"runs/{model.name}-train-{timestamp}")
writer = SummaryWriter(results_path.joinpath("logs"))
writer.add_graph(model, [i.to(current_device) for i in next(iter(train_dataloader))[:-1]])

print(f"Train id: {timestamp}")

# Train the model
epochs = 1
batch_count = len(train_dataloader)

report = "RUN REPORT\n------\n"
report += f"Train id: {timestamp}\n"
report += f"Model name: {model.name}\n"
report += f"Using {current_device} device\n"
report += f"Dataset: {dataset_id}\n"
report += f"Data instances: Train->{train_size}, Validation->{val_size}\n"
report += f"Batch size: {batch_size}\n"
report += f"Epochs: {epochs}\n"
report += f"Loss function: \n{loss_fn}\n"
report += f"Optimizer: \n{optimizer}\n"
print(report)
for i in range(epochs):
    with tqdm(total=batch_count, unit="batch", leave=False) as pbar:
        pbar.set_description(f"Epoch [{i:4d}/{epochs:4d}]")

        # Training
        model.train()
        running_loss = 0
        for j, (x, y) in enumerate(train_dataloader):
            x, y = x.to(current_device), y.to(current_device)
            predictions = model(x)
            # TODO: refactor code since model returns 6 preds now
            # TODO: write dispnet loss function so that it changes weights of the finer predictions over time
            # TODO: implement resizing for loss function
            # TODO: check which whether to crop last part of the network, that part you can make untrainable
            # TODO: try to ingest the pre-trained weights and start training on them
            pred = predictions[-1]  # finest one
            if y.shape[-2:] != pred.shape[-2:]:
                print("needs interpolating")
                pred = interpolate(pred, size=y.shape[-2:], mode="area")
            else:
                print("doesn't need")
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss = loss.item()
            running_loss += train_loss
            batch_idx = i * batch_count + j + 1
            writer.add_scalar("Loss/train", train_loss, batch_idx)
            pbar.set_postfix(loss=f"{train_loss:.4f}")
            pbar.update(True)

        # Validation
        running_val_loss = 0
        model.eval()
        with torch.no_grad():
            for k, (x, y) in enumerate(validation_dataloader):
                x, y = x.to(current_device), y.to(current_device)
                y_hat = model(x)
                running_val_loss += loss_fn(y_hat, y).item()

        avg_loss = running_loss / len(train_dataloader)  # loss per batch
        avg_val_loss = running_val_loss / len(validation_dataloader)

        writer.add_scalars('Avg Losses per Epoch',
                           {'Training': avg_loss, 'Validation': avg_val_loss},
                           i + 1)
        writer.flush()

print("Finished training!")

report += f"Model Summary:\n{model.__str__()}\n"
report += f"Trainable parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"

with open(results_path.joinpath('report.txt'), "w") as f:
    f.write(report)
torch.save(model.state_dict(), results_path.joinpath("model.pth"))
print(f"Dumped logs to {results_path}")
