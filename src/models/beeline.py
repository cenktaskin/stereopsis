from torch import nn


class NNModel(nn.Module):
    name = "beeline"

    def __init__(self):
        super(NNModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=7, stride=3, padding=3),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding='same'),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.conv4 = nn.Conv2d(128, 1, kernel_size=3, padding='same')

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x
