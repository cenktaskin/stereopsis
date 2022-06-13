from torch import nn
from collections import OrderedDict


class NNModel(nn.Module):
    def __init__(self, name, c_list):
        super(NNModel, self).__init__()
        self.name = name

        layers = OrderedDict()
        layers[f"conv_{0}"] = conv_layer(c_in=6, c_out=c_list[0], k=7, s=3, p=3)
        layers[f"conv_{1}"] = conv_layer(c_in=c_list[0], c_out=c_list[1], s=2, p=1)
        for i in range(len(c_list) - 2):
            layers[f"conv_{i + 2}"] = conv_layer(c_in=c_list[i + 1], c_out=c_list[i + 2], p="same")
        layers["conv_last"] = nn.Conv2d(c_list[-1], 1, kernel_size=3, padding="same")

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        return x


def conv_layer(c_in, c_out, k=3, s=None, p=None):
    if not s:
        s = 1
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p),
                         nn.BatchNorm2d(c_out),
                         nn.ReLU())
