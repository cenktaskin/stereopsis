import torch
import cv2
import numpy as np
from torch.nn.functional import interpolate

from report_model_comparison import runs
from src.dataset import data_path
from src.models.dispnet import NNModel
from src.preprocessing.data_io import RawDataHandler, show_images

logs = {r: data_path.joinpath(f"logs/{runs[r]}") for r in runs}

epochs = 200
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = data_path.joinpath("processed/report-imgs/")


for i, run in enumerate(logs):
    print(run)
    dh = RawDataHandler(data_dir=img_path.joinpath(str(i)), prefixes=("sl", "sr", "dp"))
    for ts, st, dp in dh.iterate_over_imgs():
        a = cv2.imread(img_path.joinpath("preds_by_disp", str(i), f"pred_{ts}.tiff").as_posix(), flags=cv2.IMREAD_UNCHANGED)
        show_images([a], row_count=1, titles=[f"{i}-{ts}"])
