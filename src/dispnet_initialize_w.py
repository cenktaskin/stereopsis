import pickle

import torch
from dataset import data_path
from models import dispnet

if False:
    with open("layer_name_conv.txt", "r") as f:
        file = f.read()

    lines = file.split("\n")[:-1]
    print(lines[-1])
    a = [line.split("-") for line in lines]
    converter = {pair[1]: pair[0] for pair in a}
    with open("layer_name_dict.pkl", "wb") as f:
        pickle.dump(converter, f)

    my_keys = list(my_state_dict.keys())
    dispnet_keys = list(state_dict.keys())[4:]

    res = ""
    for i in range(len(dispnet_keys)):
        comparison = f"{dispnet_keys[i]}-{my_keys[i]}\n"
        print(comparison, end="")
        flag = state_dict[dispnet_keys[i]].shape == my_state_dict[my_keys[i]].shape
        print(flag)
        res += comparison

with open("layer_name_dict.pkl", "rb") as f:
    layer_converter = pickle.load(f)

print(layer_converter)
exit()
state_dict = torch.load(data_path.joinpath('raw/dispnet_cvpr2016.pt'))
dispnet_keys = list(state_dict.keys())[4:]
net = dispnet.NNModel()
my_state_dict = net.state_dict()

resulting_state_dict = {}
with torch.no_grad():
    for key in dispnet_keys:
        flag = state_dict[key].shape == my_state_dict[layer_converter[key]].shape
        if flag:
            resulting_state_dict[layer_converter[key]] = state_dict[key]

print(resulting_state_dict)


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