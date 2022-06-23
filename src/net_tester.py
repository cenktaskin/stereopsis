import numpy as np
import torch

from models import dispnet
from dataset import StereopsisDataset, show_images
from loss import MaskedEPE, MultilayerSmoothL1viaPool


def tester(log_path, dataset_type, old_type_idx):
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader_idx = torch.load(log_path.joinpath("val_loader.pt")).dataset.indices
    if old_type_idx:
        val_loader_idx = np.loadtxt(log_path.joinpath("validation_indices.txt")).astype(int)

    dataset_path = log_path.parent.parent.joinpath(f"processed/dataset-20220610-{dataset_type}")
    dataset = StereopsisDataset(dataset_path)

    model_weights = torch.load(next(log_path.glob("model*")), map_location=current_device)
    model = dispnet.NNModel(batch_norm=True)
    model.load_state_dict(model_weights)

    acc_fn = MaskedEPE()
    loss_fn = MultilayerSmoothL1viaPool()
    model.eval()
    with torch.no_grad():
        for idx in val_loader_idx:
            x, y = dataset.__getitem__(idx)
            x_tensor = torch.from_numpy(x).unsqueeze(dim=0).float()
            y_tensor = torch.from_numpy(y).unsqueeze(dim=0).float()
            y_hats = model(x_tensor)
            pred = y_hats[-1].squeeze()
            loss = loss_fn(y_hats, y_tensor, stage=3)
            accuracy = acc_fn(y_hats, y_tensor)
            tit = ["left_img", "right_img", "label", "pred"]
            main_tit = f"Loss:{loss:.4f}, Accuracy:{accuracy:.4f}"
            show_images([*np.split(x.squeeze(), 2, axis=0), y, pred], titles=tit, row_count=2, main_title=main_tit)
