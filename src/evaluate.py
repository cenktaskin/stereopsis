import argparse

from net_tester import tester
from dataset import data_path

arg_parser = argparse.ArgumentParser(description="NN Evaluator")
arg_parser.add_argument("-r", "--run-name", type=str)
arg_parser.add_argument("-dt", "--dataset-type", type=str, default="fullres")
arg_parser.add_argument("-o", "--old-type-idx", action='store_true')
args = arg_parser.parse_args()

log_path = None
matching_runs = list(data_path.joinpath("logs").glob(f"*{args.run_name}*"))
if len(matching_runs) == 1:
    log_path = matching_runs[0]
else:
    if len(matching_runs) == 0:
        print("No matching run found, choose one:")
        possible_logs = list(data_path.joinpath("logs").glob("*"))
    else:
        print("More than one matching run found, choose one:")
        possible_logs = matching_runs
    for i, p in enumerate(possible_logs):
        print(f"{i}) {p.name}")
    try:
        idx = int(input("Choice: "))
        log_path = possible_logs[idx]
    except ValueError:
        exit("Unexpected input type!")
    except IndexError:
        exit("Didn't match!")

del args.run_name
tester(log_path, **vars(args))
