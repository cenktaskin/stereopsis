import argparse
import pytz
import socket
from datetime import datetime
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from loss import MultilayerSmoothL1, MultilayerSmoothL1viaPool, MaskedEPE
from dataset import StereopsisDataset, data_path
from net_trainer import trainer


def generate_report():
    return f"""
### RUN REPORT
    Timestamp: {timestamp}
    Run name: {args.run_name}
    Host name: {socket.gethostname()}
    Device: {current_device}
    Num_workers: {args.num_workers}
    Model name: {model_name}
    Pretrained: {not args.not_pretrained}
    Dataset: {dataset_id}-{args.dataset_type}
    Train samples: {train_size} 
    Validation samples: {val_size}
    Epochs: {args.epochs}
    Batch size: {args.batch_size}
    Batch norm: {args.batch_norm}
    Learning rate: {args.learning_rate}
    Scheduler step: {args.scheduler_step} 
    Scheduler gamma: {args.scheduler_gamma}
    Loss function: {loss_fn.name}
    Accuracy metric: {accuracy_fn.name}
    Optimizer: Adam
"""


arg_parser = argparse.ArgumentParser(description="NN Trainer")
arg_parser.add_argument("-e", "--epochs", type=int, default=10)
arg_parser.add_argument("-bs", "--batch-size", type=int, default=16)
arg_parser.add_argument("-bn", "--batch-norm", type=bool, default=True)
arg_parser.add_argument("-npre", "--not-pretrained", action='store_true')
arg_parser.add_argument("-lr", "--learning-rate", type=float, default=10 ** -4)
arg_parser.add_argument("-schs", "--scheduler-step", type=int, default=10)
arg_parser.add_argument("-schg", "--scheduler-gamma", type=float, default=0.5)
arg_parser.add_argument("-nw", "--num-workers", type=int, default=4)
arg_parser.add_argument("-dt", "--dataset-type", type=str, default="fullres")
arg_parser.add_argument("-n", "--run-name", type=str, default=None)
arg_parser.add_argument("-sub", "--subsample", type=int, default=False)
arg_parser.add_argument("-lf", "--loss-func-idx", type=int, default=0)
args = arg_parser.parse_args()

model_name = "dispnet"
timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
run_id = f"{model_name}-{timestamp}-{socket.gethostname()}"
if args.run_name:
    run_id += f"-{args.run_name}"
results_path = data_path.joinpath(f"logs/{run_id}")

dataset_id = "20220610"
data_split_ratio = 0.99
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-{args.dataset_type}")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = StereopsisDataset(dataset_path)
if args.subsample:
    active_samples = args.subsample
    dataset, _ = torch.utils.data.random_split(dataset, [active_samples, len(dataset) - active_samples])
train_size = int(data_split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

loss_fns = [MultilayerSmoothL1(), MultilayerSmoothL1viaPool()]
loss_fn = loss_fns[args.loss_func_idx]
accuracy_fn = MaskedEPE()

writer = SummaryWriter(results_path)

report = generate_report()
writer.add_text(run_id, report)
writer.flush()
print(report)
start_time = time.time()
del args.dataset_type, args.run_name, args.subsample, args.loss_func_idx

trainer(model_name=model_name, train_dataset=train_dataset, validation_dataset=val_dataset, loss_fn=loss_fn,
        accuracy_fn=accuracy_fn, current_device=current_device, writer=writer, **vars(args))

print(f"Finished training in {time.time() - start_time}!")
print(f"Dumped logs to {results_path}")
