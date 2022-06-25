import argparse
import torch
import numpy as np

from net_tester import tester
from dataset import data_path, StereopsisDataset

arg_parser = argparse.ArgumentParser(description="NN Evaluator")
arg_parser.add_argument("-r", "--run-name", type=str)
arg_parser.add_argument("-chdt", "--choose-dataset", action='store_true')
arg_parser.add_argument("-chw", "--choose-weights", action='store_true')
arg_parser.add_argument("-o", "--old-type-idx", action='store_true')
args = arg_parser.parse_args()


def offer_choices(options):
    for i, p in enumerate(options):
        print(f"{i}) {p.name}")
    try:
        idx = int(input("Choice: "))
        return options[idx]
    except ValueError:
        exit("Unexpected input type!")
    except IndexError:
        exit("Didn't match!")


possible_runs = list(data_path.joinpath("logs").glob(f"*{args.run_name}*"))
if len(possible_runs) == 1:
    log_path = possible_runs[0]
else:
    if len(possible_runs) == 0:
        print("No matching run found, checking parent dir:")
        possible_runs = list(data_path.joinpath("logs").glob("*"))
    else:
        print("More than one matching run found, choose one:")
    log_path = offer_choices(possible_runs)

possible_sets = list(log_path.parent.parent.glob(f"processed/dataset-*"))
if not args.choose_dataset:
    dataset_path = possible_sets[-1]
else:
    print(f"Choose a dataset:")
    dataset_path = offer_choices(possible_sets)

possible_weights = list(log_path.glob("model*.pt"))
if len(possible_weights) == 0:
    weight_path = None
    exit("No matching weights found, exiting...")
elif len(possible_weights) == 1:
    weight_path = possible_weights[0]
else:
    if not args.choose_weights:
        weight_path = possible_weights[-1]
    else:
        print("More than one weight file is found, choose one:")
        weight_path = offer_choices(possible_weights)

val_loader_idx = torch.load(log_path.joinpath("val_loader.pt"))
print(val_loader_idx)
# val_loader_idx.dataset = StereopsisDataset(dataset_path)
# for j, (x, y) in enumerate(val_loader_idx):
#    print(j)
# exit()
# .dataset.indices
if args.old_type_idx:
    val_loader_idx = np.loadtxt(log_path.joinpath("validation_indices.txt")).astype(int)

dataset = StereopsisDataset(dataset_path)

tester(weight_path, dataset, val_loader_idx)
