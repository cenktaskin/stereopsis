import argparse
import pytz
import socket
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import StereopsisDataset, data_path
from net_trainer import trainer

arg_parser = argparse.ArgumentParser(description="NN Trainer")
arg_parser.add_argument("-e", "--epochs", type=int, default=10)
arg_parser.add_argument("-bs", "--batch-size", type=int, default=16)
arg_parser.add_argument("-bn", "--batch-norm", type=bool, default=True)
arg_parser.add_argument("-npre", "--not-pretrained", action='store_true', default=False)
arg_parser.add_argument("-lr", "--learning-rate", type=float, default=10 ** -4)
arg_parser.add_argument("-dt", "--dataset-type", type=str, default="fullres")
arg_parser.add_argument("-schg", "--scheduler-gamma", type=float, default=0.5)
arg_parser.add_argument("-schs", "--scheduler-step", type=int, default=10)
arg_parser.add_argument("-n", "--run-name", type=str, default=None)
args = arg_parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
batch_norm = args.batch_norm
pretrained = not args.not_pretrained
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
results_path = data_path.joinpath(f"logs/{run_id}")

dataset_id = "20220610"
data_split_ratio = 0.99
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-{dataset_type}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = StereopsisDataset(dataset_path)
# active_samples = 1000#  to activate less part of it
# dataset, _ = torch.utils.data.random_split(dataset, [active_samples, len(dataset)-active_samples])
train_size = int(data_split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

writer = SummaryWriter(results_path)

report = f"""
### RUN REPORT
    Timestamp: {timestamp}
    Run name: {run_name}
    Host name: {socket.gethostname()}
    Device: {current_device}
    Model name: {model_name}
    Pretrained: {pretrained}
    Dataset: {dataset_id}-{dataset_type}
    Train samples: {train_size} 
    Validation samples: {val_size}
    Epochs: {epochs}
    Batch size: {batch_size}
    Learning rate: {learning_rate}
    Scheduler step:{scheduler_step} 
    Scheduler gamma: {scheduler_gamma}
    Loss function: MultiLayerSmoothL1
    Accuracy metric: MaskedEPE
    Optimizer: Adam
"""

print(report)
writer.add_text(run_id, report)
writer.flush()

trainer(model_name=model_name, train_dataset=train_dataset, validation_dataset=val_dataset,
        current_device=current_device, epochs=epochs, batch_size=batch_size, batch_norm=batch_norm,
        pretrained=pretrained, learning_rate=learning_rate, scheduler_step=scheduler_step,
        scheduler_gamma=scheduler_gamma, writer=writer)

print("Finished training!")
print(f"Dumped logs to {results_path}")
