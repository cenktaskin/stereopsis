from torch import nn


class NeuralNetworkModel(nn.Module):
    name = "beeline"

    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 3, stride=3, padding=1),
                                   nn.BatchNorm2d(3),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3, 3, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(3),
                                   nn.ReLU())
        self.conv3 = nn.Conv2d(3, 1, 3, padding='same')

    def forward(self, l, r):
        l = l.float()
        r = r.float()
        # l_img, r_img = split(x, x.shape[2] // 2, dim=2)

        x1 = self.conv1(l)

        x2 = self.conv1(r)

        stacked = x1 + x2
        y = self.conv2(stacked)
        return y
