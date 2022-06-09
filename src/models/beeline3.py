from torch import nn


class NNModel(nn.Module):
    name = "beeline3"

    def __init__(self):
        super(NNModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=7, stride=3, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding='same'),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding='same'),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding='same'),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU())
        self.conv6 = nn.Conv2d(1024, 1, kernel_size=3, padding='same')

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)


        return x
