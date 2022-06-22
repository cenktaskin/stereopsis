import torch
from models import dispnet
from dataset import data_path, StereopsisDataset, show_images
import numpy as np
from loss import MaskedEPE

current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_name = "dispnet-202206222310-namo-LossFComparisonOldOne"
log_path = data_path.joinpath(f"logs/{run_name}")

val_loader = torch.load(log_path.joinpath("val_loader.pt"))
#val_loader_idx = np.loadtxt(log_path.joinpath("validation_indices.txt")).astype(int)

dataset_path = data_path.joinpath(f"processed/dataset-20220610-origres")
dataset = StereopsisDataset(dataset_path)

model_weights = torch.load(next(log_path.glob("model*")), map_location=current_device)
model = dispnet.NNModel(batch_norm=True)
model.load_state_dict(model_weights)

acc_fn = MaskedEPE()
model.eval()
with torch.no_grad():
    for idx in val_loader.dataset.indices:
    #for idx in val_loader_idx:
        x, y = dataset.__getitem__(idx-10)
        x_tensor = torch.from_numpy(x).unsqueeze(dim=0).float()
        y_hats = model(x_tensor)
        pred = y_hats[-1].squeeze()
        print(acc_fn(y_hats,torch.from_numpy(y).unsqueeze(dim=0).float()))
        tit = ["left_img", "right_img", "label", "pred"]
        show_images([*np.split(x.squeeze(), 2, axis=0), y, pred], titles=tit, row_count=2)

