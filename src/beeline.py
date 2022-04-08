from torch import nn, split


class BeelineModel(nn.Module):
    def __init__(self):
        super(BeelineModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.b_norm1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 1, 3, stride=3, padding=1)

    def forward(self, x):
        l_img, r_img = split(x, x.shape[2]//2, dim=2)

        x1 = self.conv1(l_img)
        x1 = self.b_norm1(x1)
        x1 = nn.ReLU(x1)

        x2 = self.conv1(r_img)
        x2 = self.b_norm1(x2)
        x2 = nn.ReLU(x2)

        stacked = x1+x2
        y = self.conv2(stacked)
        return y
