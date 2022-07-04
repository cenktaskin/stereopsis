import torch

from net_tester import tester
from dataset import data_path, StereopsisDataset
from models.dispnet import NNModel


runs = {"vanilla": "202206271316-namo-FullRes200ELowLR",
        "undistorted": "202206290926-namo",
        "rectified": "202206271319-irmo-Rectified200ELowLR",
        "registered": "202206271548-phisrv-200ELowLRNewWeights",
        "registered+rectified": "202206290846-phisrv"}

logs = {r: data_path.joinpath(f"logs/{runs[r]}") for r in runs}
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_path = data_path.joinpath("processed/20220701-predictions")

for run in logs:
    model = NNModel(batch_norm=True)
    ds = StereopsisDataset(data_path.joinpath(f"processed/dataset-20220701test-{run}"), val_split_ratio=1.00)
    for w in logs[run].glob("model-e*.pt"):
        model_weights = torch.load(w, map_location=current_device)
        model.load_state_dict(model_weights)
        results = tester(model, ds)
        loss = 0
        acc = 0
        for ts in results:
            loss += results[ts]['loss']
            acc += results[ts]['accuracy']
        print(f"Run: {run}, epochs: {w.name}")
        print(f"Loss {loss/len(results)}")
        print(f"Acc {acc/len(results)}")
