import pickle

import torch
from dataset import data_path
from models import dispnet

target_weights = torch.load(data_path.joinpath('raw/dispnet_cvpr2016.pt'))

with open("layer_name_dict.pkl", "rb") as f:
    layer_converter = pickle.load(f)

net = dispnet.NNModel()

for name, param in net.named_parameters():
    param.data = target_weights[layer_converter[name]]


torch.save(net.state_dict(), 'dispnet_weights.pth')
