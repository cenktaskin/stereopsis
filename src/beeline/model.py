from torch import nn, split


class BeelineModel(nn.Module):
    name = "beeline"

    def __init__(self):
        super(BeelineModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 3, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(3),
                                   nn.ReLU())
        self.conv2 = nn.Conv2d(3, 1, 3, stride=3, padding=1)

    def forward(self, l, r):
        l = l.float()
        r = r.float()
        # l_img, r_img = split(x, x.shape[2] // 2, dim=2)

        x1 = self.conv1(l)

        x2 = self.conv1(r)

        stacked = x1 + x2
        y = self.conv2(stacked)
        return y


class BeelineModel2(nn.Module):
    name = "beeline2"

    def __init__(self):
        super(BeelineModel2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU())
        self.conv2 = nn.Conv2d(12, 1, 3, stride=3, padding=1)

    def forward(self, l, r):
        l = l.float()
        r = r.float()
        # l_img, r_img = split(x, x.shape[2] // 2, dim=2)

        x1 = self.conv1(l)

        x2 = self.conv1(r)

        stacked = x1 + x2
        y = self.conv2(stacked)
        return y
