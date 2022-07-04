import argparse
import torch

from net_tester import tester
from dataset import data_path
from models.dispnet import NNModel

arg_parser = argparse.ArgumentParser(description="NN Evaluator")
arg_parser.add_argument("-r", "--run-name", type=str)
arg_parser.add_argument("-chw", "--choose-weights", action='store_true')
arg_parser.add_argument("-bn", "--batch-norm", type=bool, default=True)
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
        possible_runs = sorted(list(data_path.joinpath("logs").glob("*")))
    else:
        print("More than one matching run found, choose one:")
    log_path = offer_choices(possible_runs)

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

current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load(log_path.joinpath("dataset.pt"))
dataset.assert_img_dir()

model = NNModel(batch_norm=args.batch_norm)
model_weights = torch.load(weight_path, map_location=current_device)
model.load_state_dict(model_weights)

tester(model, dataset, log_path.name, weight_path.name, view_results=True)
