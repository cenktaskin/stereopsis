import torch
import cv2
import numpy as np
from torch.nn.functional import interpolate

from report_model_comparison import runs
from src.dataset import data_path
from src.models.dispnet import NNModel
from src.preprocessing.data_io import RawDataHandler

logs = {r: data_path.joinpath(f"logs/{runs[r]}") for r in runs}

epochs = 200
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = data_path.joinpath("processed/report-imgs/")

for i, run in enumerate(logs):

    model = NNModel(batch_norm=True)
    model_weights = torch.load(logs[run].joinpath(f"model-e{epochs}.pt"), map_location=current_device)
    model.load_state_dict(model_weights)
    model.eval()

    dh = RawDataHandler(data_dir=img_path.joinpath(str(i)), prefixes=("sl", "sr", "dp"))
    with torch.no_grad():
        for ts, st, dp in dh.iterate_over_imgs():
            img_l, img_r = dh.parse_stereo_img(st)

            x = np.concatenate([img_l, img_r], axis=2).transpose((2, 0, 1))
            x = torch.from_numpy(x).unsqueeze(dim=0)
            y = torch.from_numpy(dp)

            y_hat = model(x.float())
            pred = interpolate(y_hat[-1], size=y.shape[-2:], mode="bilinear").squeeze().numpy()
            cv2.imwrite(img_path.joinpath("preds_by_disp", str(i), f"pred_{ts}.tiff").as_posix(), pred)
