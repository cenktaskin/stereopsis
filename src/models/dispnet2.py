from torch import nn, cat
from torch.nn.functional import leaky_relu


# for 180x320
class NNModel(nn.Module):
    name = 'dispnet'

    def __init__(self):
        super(NNModel, self).__init__()

        self.conv1a = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=(1, 2))
        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=(2, 1))
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(2, 1))
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=(1, 2))
        self.conv5b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6a = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv6b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)

        self.pr6 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.up6to5 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.iconv5 = nn.Conv2d(in_channels=1025, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pr5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.up5to4 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=(1, 2))
        self.upconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=(1, 2))
        self.iconv4 = nn.Conv2d(in_channels=769, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pr4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.up4to3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=(2, 1))
        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=(2, 1))
        self.iconv3 = nn.Conv2d(in_channels=385, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pr3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.up3to2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=(1, 0))
        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=(1, 0))
        self.iconv2 = nn.Conv2d(in_channels=193, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pr2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.up2to1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=(0, 1))
        self.upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=(0, 1))
        self.iconv1 = nn.Conv2d(in_channels=97, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pr1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x00):
        print(f"{x00.shape=}")
        x1a = leaky_relu(self.conv1a(x00), 0.1)
        print(f"{x1a.shape=}")
        x2a = leaky_relu(self.conv2a(x1a), 0.1)
        print(f"{x2a.shape=}")
        x3a = leaky_relu(self.conv3a(x2a), 0.1)
        print(f"{x3a.shape=}")
        x3b = leaky_relu(self.conv3b(x3a), 0.1)
        print(f"{x3b.shape=}")
        x4a = leaky_relu(self.conv4a(x3b), 0.1)
        print(f"{x4a.shape=}")
        x4b = leaky_relu(self.conv4b(x4a), 0.1)
        print(f"{x4b.shape=}")
        x5a = leaky_relu(self.conv5a(x4b), 0.1)
        print(f"{x5a.shape=}")
        x5b = leaky_relu(self.conv5b(x5a), 0.1)
        print(f"{x5b.shape=}")
        x6a = leaky_relu(self.conv6a(x5b), 0.1)
        print(f"{x6a.shape=}")
        x6b = leaky_relu(self.conv6b(x6a), 0.1)
        print(f"{x6b.shape=}")

        pr6 = self.pr6(x6b)
        print(f"{pr6.shape=}")

        ux5 = leaky_relu(self.upconv5(x6b), 0.1)
        print(f"{ux5.shape=}")
        ix5 = self.iconv5(cat([ux5, self.up6to5(pr6), x5b], 1))
        print(f"{ix5.shape=}")
        pr5 = self.pr5(ix5)
        print(f"{pr5.shape=}")

        ux4 = leaky_relu(self.upconv4(ix5), 0.1)
        print(f"{ux4.shape=}")
        ix4 = self.iconv4(cat([ux4, self.up5to4(pr5), x4b], 1))
        print(f"{ix4.shape=}")
        pr4 = self.pr4(ix4)
        print(f"{pr4.shape=}")

        ux3 = leaky_relu(self.upconv3(ix4), 0.1)
        print(f"{ux3.shape=}")
        ix3 = self.iconv3(cat([ux3, self.up4to3(pr4), x3b], 1))
        print(f"{ix3.shape=}")
        pr3 = self.pr3(ix3)
        print(f"{pr3.shape=}")

        ux2 = leaky_relu(self.upconv2(ix3), 0.1)
        print(f"{ux2.shape=}")
        ix2 = self.iconv2(cat([ux2, self.up3to2(pr3), x2a], 1))
        print(f"{ix2.shape=}")
        pr2 = self.pr2(ix2)
        print(f"{pr2.shape=}")

        ux1 = leaky_relu(self.upconv1(ix2), 0.1)
        print(f"{ux1.shape=}")
        ix1 = self.iconv1(cat([ux1, self.up2to1(pr2), x1a], 1))
        print(f"{ix1.shape=}")
        pr1 = self.pr1(ix1)
        print(f"{pr1.shape=}")

        return pr6, pr5, pr4, pr3, pr2, pr1
