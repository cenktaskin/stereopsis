from datetime import datetime
import pytz
from tqdm import tqdm
import importlib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import LabelTransformer, StereopsisDataset, np_to_tensor, data_path
from loss import MaskedMSE

dataset_id = "20220301"
dataset_path = data_path.joinpath(f"raw/dataset-{dataset_id}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8
data_split_ratio = 0.95
label_transformer = LabelTransformer(h=120, w=214)
dataset = StereopsisDataset(dataset_path, transform=transforms.Compose([np_to_tensor]),
                            target_transform=transforms.Compose([label_transformer]))

train_size = int(data_split_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# for layer_count in range(2,5):
layer_count = 2
channel_list = [2 ** (i + 6) for i in range(layer_count)]
model_type = "beeline"
model_net = getattr(importlib.import_module(f"models.{model_type}"), "NNModel")
model = model_net(f"beeline{len(channel_list)}", channel_list).to(current_device)

loss_fn = MaskedMSE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
results_path = data_path.joinpath(f"runs/{model.name}-train-{timestamp}")
writer = SummaryWriter(results_path.joinpath("logs"))
writer.add_graph(model, [i.to(current_device) for i in next(iter(train_dataloader))[:-1]])

print(f"Train id: {timestamp}")
print(f"Using {current_device} device")
print(f"Batch size: {batch_size}")
print(f"Sample size: Train: {train_size}, Test: {test_size}")

# Train the model
epochs = 1
batch_count = len(train_dataloader)
for i in range(epochs):
    with tqdm(total=batch_count, unit="batch", leave=False) as pbar:
        pbar.set_description(f"Epoch [{i:4d}/{epochs:4d}]")

        # Training
        model.train()
        running_loss = 0
        for j, (x, y) in enumerate(train_dataloader):
            x, y = x.to(current_device), y.to(current_device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

report = "RUN REPORT\n------\n"
report += f"Train id: {timestamp}"
report += f"Using {current_device} device"
report += f"Dataset:{dataset_id}\n"
report += f"Data instances: Train->{train_size}, Test->{test_size}"
report += f"Batch size: {batch_size}"
report += f"Epochs: {epochs}"
report += f"Loss function: {loss_fn}"
report += f"Optimizer: {optimizer}"
report += f"Model Summary:\n{model.__str__()}"
report += f"Trainable parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"

with open(results_path.joinpath('report.txt'), "w") as f:
    f.write(report)
torch.save(model.state_dict(), results_path.joinpath("model.pth"))
print("Finished training!")
