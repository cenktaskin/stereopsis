import argparse
import pytz
import socket
from datetime import datetime
import time
from importlib import import_module

import torch
from torch.utils.tensorboard import SummaryWriter

from loss import MultilayerSmoothL1, MultilayerSmoothL1viaPool, MaskedEPE
from dataset import StereopsisDataset, data_path
from net_trainer import trainer
from pretrained_weights import fetch_pretrained_dispnet_weights


def generate_report():
    return f"""
### REPORT
#### Run
    Name: {args.run_name}
    Timestamp: {timestamp}
    Host: {socket.gethostname()}
    Device: {'cuda' if torch.cuda.is_available() else 'cpu'}
    Num_workers: {args.num_workers}
#### Model
    Name: {model_name}
    Pretrained: {not args.not_pretrained}
    Encoder frozen: {args.freeze_encoder}
    Loss function: {loss_fn.name}
    Accuracy metric: {accuracy_fn.name}
#### Data
    Dataset: {dataset_id}-{args.dataset_type}
    #Samples train: {len(dataset.train_idx)} 
             val: {len(dataset.val_idx)}
#### Hparams
    Epochs: {args.epochs}
    Batch size: {args.batch_size}
    Batch norm: {args.batch_norm}
    Optimizer: Adam
    Learning rate: {args.learning_rate}
    Scheduler step: {args.scheduler_step} 
    Scheduler gamma: {args.scheduler_gamma} 
"""


arg_parser = argparse.ArgumentParser(description="NN Trainer")
arg_parser.add_argument("-e", "--epochs", type=int, default=10)
arg_parser.add_argument("-bs", "--batch-size", type=int, default=16)
arg_parser.add_argument("-lr", "--learning-rate", type=float, default=10 ** -4)
arg_parser.add_argument("-schs", "--scheduler-step", type=int, default=10)
arg_parser.add_argument("-schg", "--scheduler-gamma", type=float, default=0.5)
arg_parser.add_argument("-nw", "--num-workers", type=int, default=4)
arg_parser.add_argument("-bn", "--batch-norm", type=bool, default=True)
arg_parser.add_argument("-np", "--not-pretrained", action='store_true')
arg_parser.add_argument("-dt", "--dataset-type", type=str, default="fullres")
arg_parser.add_argument("-n", "--run-name", type=str, default=None)
arg_parser.add_argument("-sub", "--subsample", type=float, default=1.0)
arg_parser.add_argument("-lf", "--loss-func-idx", type=int, default=1)
arg_parser.add_argument("-fe", "--freeze-encoder", action='store_true')
args = arg_parser.parse_args()

model_name = "dispnet"
timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
run_id = f"{timestamp}-{socket.gethostname()}"
if args.run_name:
    run_id += f"-{args.run_name}"

dataset_id = "20220610"
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-{args.dataset_type}")
dataset = StereopsisDataset(dataset_path, val_split_ratio=0.01, subsample_ratio=args.subsample)
loss_fns = [MultilayerSmoothL1(), MultilayerSmoothL1viaPool()]
loss_fn = loss_fns[args.loss_func_idx]
accuracy_fn = MaskedEPE()

model_net = getattr(import_module(f"models.{model_name}"), "NNModel")
model = model_net(args.batch_norm)
if not args.not_pretrained:
    model.load_state_dict(fetch_pretrained_dispnet_weights(model))
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)

results_path = data_path.joinpath(f"logs/{run_id}")
results_path.mkdir()
torch.save(dataset, results_path.joinpath("dataset.pt"))
writer = SummaryWriter(results_path)
report = generate_report()
writer.add_text(run_id, report)
writer.flush()
writer.add_graph(model, torch.randn((1, 6, 384, 768), requires_grad=False))

print(report)
start_time = time.time()

trainer(model=model, dataset=dataset, loss_fn=loss_fn, accuracy_fn=accuracy_fn, writer=writer, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma, num_workers=args.num_workers)

print(f"Finished training in {time.time() - start_time}!")
print(f"Dumped logs to {results_path}")
