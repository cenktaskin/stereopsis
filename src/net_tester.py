import numpy as np
import torch

from models import dispnet
from dataset import show_images
from loss import MaskedEPE, MultilayerSmoothL1viaPool


def tester(weights_path, dataset, val_idx):
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = dispnet.NNModel(batch_norm=True)
    model_weights = torch.load(weights_path, map_location=current_device)
    model.load_state_dict(model_weights)

    acc_fn = MaskedEPE()
    loss_fn = MultilayerSmoothL1viaPool()
    model.eval()
    with torch.no_grad():
        for idx in val_idx:
            sample, label = dataset.__getitem__(idx)
            x_tensor = torch.from_numpy(sample).unsqueeze(dim=0).float()
            y_tensor = torch.from_numpy(label).unsqueeze(dim=0).float()
            y_hats = model(x_tensor)
            loss = loss_fn(y_hats, y_tensor, stage=3)
            accuracy = acc_fn(y_hats, y_tensor)
            tit = ["left_img", "right_img", "label", "pred"]
            prediction = y_hats[-1].numpy().squeeze()
            print(sample.shape)
            show_images([*np.split(sample.squeeze(), 2, axis=0), label, prediction], titles=tit, row_count=2,
                        main_title=f"{weights_path.name} \n Loss:{loss:.4f}, EPE:{accuracy:.4f}")
