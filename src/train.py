import argparse
import pytz
import socket
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import StereopsisDataset, data_path
from model_trainer import model_trainer

arg_parser = argparse.ArgumentParser(description="NN Trainer")
arg_parser.add_argument("-e", "--epochs", type=int, default=10)
arg_parser.add_argument("-bs", "--batch_size", type=int, default=16)
arg_parser.add_argument("-bn", "--batch_norm", type=bool, default=True)
arg_parser.add_argument("-pre", "--pretrained", type=bool, default=True)
arg_parser.add_argument("-lr", "--learning_rate", type=float, default=10 ** -4)
arg_parser.add_argument("-dt", "--dataset_type", type=str, default="fullres")
arg_parser.add_argument("-schg", "--scheduler-gamma", type=float, default=0.5)
arg_parser.add_argument("-schs", "--scheduler-step", type=int, default=10)
arg_parser.add_argument("-n", "--run_name", type=str, default=None)
args = arg_parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
batch_norm = args.batch_norm
pretrained = args.pretrained
learning_rate = args.learning_rate
dataset_type = args.dataset_type
scheduler_step = args.scheduler_step
scheduler_gamma = args.scheduler_gamma
run_name = args.run_name

model_name = "dispnet"
timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
run_id = f"{model_name}-{timestamp}-{socket.gethostname()}"
if run_name:
    run_id += f"-{run_name}"
results_path = data_path.joinpath(f"runs/{run_id}")

dataset_id = "20220610"
data_split_ratio = 0.99
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-{dataset_type}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = StereopsisDataset(dataset_path)
train_size = int(data_split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

writer = SummaryWriter(results_path.joinpath("logs"))

report = f"""RUN REPORT
    Model name: {model_name} pretrained:{pretrained} <br>
    Timestamp: {timestamp} <br>
    Run name: {run_name} <br>
    Using {current_device} device <br>
    Dataset: {dataset_id}-{dataset_type} <br>
    Samples train:{train_size} validation:{val_size} <br>
    Epochs: {epochs} <br>
    Batch size: {batch_size} <br>
    Learning rate: {learning_rate} <br>
    Scheduler step:{scheduler_step} Gamma: {scheduler_gamma}<br>
    Loss function: MultiLayerSmoothL1 <br>
    Accuracy metric: MaskedEPE <br>
    Optimizer: Adam <br>"""

print(report.replace("<br>", ""))
writer.add_text(run_id, report)

model, val_loader = model_trainer(model_name=model_name, train_dataset=train_dataset,
                                  validation_dataset=val_dataset, current_device=current_device,
                                  epochs=epochs, batch_size=batch_size, batch_norm=batch_norm,
                                  pretrained=pretrained, learning_rate=learning_rate, scheduler_step=scheduler_step,
                                  scheduler_gamma=scheduler_gamma, writer=writer)

torch.save(model.state_dict(), results_path.joinpath(f"model-e{epochs}.pth"))
torch.save(val_loader, results_path.joinpath("val_loader.pt"))

print("Finished training!")
print(f"Dumped logs to {results_path}")
