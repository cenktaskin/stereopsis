import pandas as pd
import torch

from src.net_tester import tester
from src.dataset import data_path, StereopsisDataset
from src.models.dispnet import NNModel

runs = {"vanilla": "202207041736-phisrv-VanillaRun",
        "undistorted": "202206290926-namo",
        "rectified": "202206271319-irmo-Rectified200ELowLR",
        "registered": "202206271548-phisrv-200ELowLRNewWeights",
        "registered+rectified": "202206290846-phisrv"}
logs = {r: data_path.joinpath(f"logs/{runs[r]}") for r in runs}

if __name__ == "__main__":
    epochs = list(range(20, 220, 20))
    acc = pd.DataFrame(0, index=epochs, columns=list(runs.keys()))
    loss = pd.DataFrame(0, index=epochs, columns=list(runs.keys()))
    for e in epochs:
        current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output_path = data_path.joinpath("processed/20220701-predictions")

        for run in logs:
            model = NNModel(batch_norm=True)
            ds = StereopsisDataset(data_path.joinpath(f"processed/dataset-20220701test-{run}"), val_split_ratio=1.00)
            model_weights = torch.load(logs[run].joinpath(f"model-e{e}.pt"), map_location=current_device)
            model.load_state_dict(model_weights)
            results = tester(model, ds, run, f"{e}epochs", view_results=False)
            loss.at[e, run] = sum([results[k]["loss"].item() for k in results]) / len(ds)
            acc.at[e, run] = sum([results[k]["accuracy"].item() for k in results]) / len(ds)
            print(loss)
            print(acc)
    loss.to_csv("test_loss.csv")
    acc.to_csv("test_acc.csv")
