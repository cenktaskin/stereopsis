import numpy as np
from report_model_comparison import runs
from src.preprocessing.data_io import data_path
from src.preprocessing.data_preprocessor import ImageResizer
import re
import torch
from torch.nn.functional import interpolate
from src.models.dispnet import NNModel
import cv2
from src.loss import MaskedEPE, MultilayerSmoothL1viaPool


# from https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NNModel(batch_norm=True)

mb_path = data_path.joinpath("raw/middlebury2014")

f = 3997.684
baseline = 193.001

image_l = cv2.cvtColor(cv2.imread(mb_path.joinpath("piano/im0.png").as_posix()), cv2.COLOR_BGR2RGB)
image_r = cv2.cvtColor(cv2.imread(mb_path.joinpath("piano/im1.png").as_posix()), cv2.COLOR_BGR2RGB)
label, _ = read_pfm(mb_path.joinpath("piano/disp1.pfm").as_posix())
label = (label + 131.111) ** -1 * f * baseline / 1000

resizer = ImageResizer(target_size=(384, 768))
image_l = resizer(image_l)
image_r = resizer(image_r)
label = resizer(label)

x = np.concatenate([image_l, image_r], axis=2).transpose((2, 0, 1))
x = torch.from_numpy(x).unsqueeze(dim=0)
y = torch.from_numpy(label)

acc_fn = MaskedEPE()
loss_fn = MultilayerSmoothL1viaPool()
print(loss_fn.multiScales)
for run in runs:
    log = data_path.joinpath(f"logs/{runs[run]}")
    model_weights = torch.load(log.joinpath(f"model-e200.pt"), map_location=current_device)
    model.load_state_dict(model_weights)

    model.eval()
    with torch.no_grad():
        y_hat = model(x.float())
        # pred = interpolate(y_hat[-1], size=y.shape[-2:], mode="bilinear").squeeze().numpy()
        # cv2.imwrite(data_path.joinpath(f"processed/report-imgs/mbury/pred_{run}.tiff").as_posix(), pred)
        print(type(y_hat[-1]))
        print(y_hat[-1].shape)
        loss = loss_fn(y_hat, y.unsqueeze(dim=0), stage=-1)
        accuracy = acc_fn(y_hat, y.unsqueeze(dim=0))
        print(run)
        print(loss)
        print(accuracy)
