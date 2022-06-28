import numpy as np
import torch
from torch.nn.functional import interpolate

from preprocessing.data_io import show_images
from loss import MaskedEPE, MultilayerSmoothL1viaPool, MultilayerSmoothL1


def tester(model, dataset, run_name, weights_name):
    _, val_loader = dataset.create_loaders(batch_size=1)

    acc_fn = MaskedEPE()
    loss_fn = MultilayerSmoothL1viaPool()

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            y_hat = model(x.float())
            loss = loss_fn(y_hat, y, stage=-1)
            accuracy = acc_fn(y_hat, y)
            tit = ["left_img", "right_img", "label", "pred"]
            pred = interpolate(y_hat[-1], size=y.shape[-2:], mode="bilinear")
            show_images([*np.split(x.squeeze(), 2, axis=0), y, pred], titles=tit, row_count=2,
                        main_title=f"{run_name}-{weights_name} \n Loss:{loss:.4f}, EPE:{accuracy:.4f}")
