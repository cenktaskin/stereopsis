from datetime import datetime
import pytz
from tqdm import tqdm
import importlib
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import StereopsisDataset, data_path
from loss import MultilayerSmoothL1, MaskedEPE
from dispnet_initialize_w import ingest_weights_to_model


epochs = 10
batch_size = 32
data_split_ratio = 0.99
batch_norm = True
pretrained = True

dataset_id = "20220610"
dataset_type = "origres"
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-{dataset_type}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = StereopsisDataset(dataset_path)
train_size = int(data_split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
train_batch_count = len(train_dataloader)

model_type = "dispnet"
model_net = getattr(importlib.import_module(f"models.{model_type}"), "NNModel")
model = model_net(batch_norm)
if pretrained:
    ingest_weights_to_model(model)
model = model.to(current_device)

loss_fn = MultilayerSmoothL1()
accuracy_fn = MaskedEPE()
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -4)  # was 0.05 on original paper but it is exploding
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
results_path = data_path.joinpath(f"runs/{model.name}-train-{timestamp}")
writer = SummaryWriter(results_path.joinpath("logs"))
writer.add_graph(model, next(iter(train_dataloader))[0].to(current_device))

np.savetxt(results_path.joinpath('validation_indices.txt'), validation_dataset.indices)

print(f"Train id: {timestamp}")

# Train the model

report = "RUN REPORT\n------\n"
report += f"Train id: {timestamp}\n"
report += f"Model name: {model.name}\n"
report += f"Pretrained: {pretrained}\n"
report += f"Using {current_device} device\n"
report += f"Dataset: {dataset_id}\n"
report += f"Data instances: Train->{train_size}, Validation->{val_size}\n"
report += f"Batch size: {batch_size}\n"
report += f"Batch normalisation: {batch_norm}\n"
report += f"Epochs: {epochs}\n"
report += f"Loss function: {loss_fn.name}\n"
report += f"Optimizer: {optimizer}\n"
print(report)
report += f"Model Summary:\n{model.__str__()}\n"
report += f"Trainable parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
with open(results_path.joinpath('report.txt'), "w") as f:
    f.write(report)
each_round = epochs // 4
for i in range(epochs):
    with tqdm(total=train_batch_count, unit="batch", leave=False) as pbar:
        pbar.set_description(f"Epoch [{i:4d}/{epochs:4d}]")

        # Training
        model.train()
        running_train_epe = 0
        for j, (x, y) in enumerate(train_dataloader):
            batch_idx = i * len(train_dataloader) + j + 1

            x, y = x.to(current_device), y.to(current_device)
            predictions = model(x)
            loss = loss_fn(predictions, y, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            running_train_epe += accuracy_fn(predictions, y).item()

            writer.add_scalar("Loss/train", train_loss, batch_idx)
            pbar.set_postfix(loss=f"{train_loss:.4f}")
            pbar.update(True)

        scheduler.step()

        # Validation
        running_val_epe = 0
        model.eval()
        with torch.no_grad():
            for k, (x, y) in enumerate(validation_dataloader):
                x, y = x.to(current_device), y.to(current_device)
                predictions = model(x)
                running_val_epe += accuracy_fn(predictions, y).item()

        avg_train_epe = running_train_epe / train_batch_count  # loss per batch
        avg_val_epe = running_val_epe / len(validation_dataloader)

        writer.add_scalars('Learning Curve [EPE/Epoch]',
                           {'Training': avg_train_epe, 'Validation': avg_val_epe},
                           i + 1)
        writer.flush()

    if i % each_round == each_round - 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # save every round
        torch.save(model.state_dict(), results_path.joinpath(f"model-e{i + 1}.pth"))

print("Finished training!")
print(f"Dumped logs to {results_path}")
