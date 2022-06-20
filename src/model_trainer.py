from datetime import datetime
import pytz
from tqdm import tqdm
import socket
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from dataset import StereopsisDataset, data_path
from loss import MultilayerSmoothL1, MaskedEPE
from dispnet_initialize_w import ingest_weights_to_model


arg_parser = argparse.ArgumentParser(description="NN Trainer")
arg_parser.add_argument("-e", "--epochs", type=int, default=10)
arg_parser.add_argument("-bs", "--batch_size", type=int, default=16)
arg_parser.add_argument("-bn", "--batch_norm", type=bool, default=True)
arg_parser.add_argument("-pre", "--pretrained", type=bool, default=True)
arg_parser.add_argument("-lr", "--learning-rate", type=float, default=10 ** -4)
args = arg_parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
batch_norm = args.batch_norm
pretrained = args.pretrained
l_rate = args.learning_rate
print(epochs)
print(batch_size)
print(batch_norm)
exit()
dataset_id = "20220610"
dataset_type = "fullres"
data_split_ratio = 0.99
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-{dataset_type}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = StereopsisDataset(dataset_path)
train_size = int(data_split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
train_batch_count = len(train_dataloader)

model = models.dispnet.NNModel(batch_norm)
if pretrained:
    ingest_weights_to_model(model)
model = model.to(current_device)

loss_fn = MultilayerSmoothL1()
accuracy_fn = MaskedEPE()
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)  # was 0.05 on original paper but it is exploding
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # add this to hparams

timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
run_id = f"{model.name}-{timestamp}-{socket.gethostname()}"
results_path = data_path.joinpath(f"runs/{run_id}")
writer = SummaryWriter(results_path.joinpath("logs"))
writer.add_graph(model, next(iter(train_dataloader))[0].to(current_device))

report = f"""RUN REPORT
Train id: {timestamp} <br>
Using {current_device} device <br>
Model name: {model.name} <br>
Dataset: {dataset_id} <br>
Data instances: Train->{train_size}, Validation->{val_size} <br>
#hparams
Loss function: {loss_fn.name} <br>
Accuracy metric: {accuracy_fn.name} <br>
Optimizer: {optimizer.__str__().split(" ")[0]} <br>
Trainable parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"""

print(report.replace("<br>", ""))
writer.add_text(run_id, report)

with open(results_path.joinpath('report.txt'), "w") as f:
    f.write(report)
# each_round = epochs // 4
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

torch.save(model.state_dict(), results_path.joinpath(f"model-e{epochs}.pth"))
torch.save(validation_dataloader, results_path.joinpath("val_loader.pt"))

print("Finished training!")
print(f"Dumped logs to {results_path}")
