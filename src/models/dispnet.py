from torch import nn, cat


class NNModel(nn.Module):
    name = 'dispnet'

    def __init__(self, batch_norm=True, verbose=False):
        super(NNModel, self).__init__()
        self.verbose = verbose

        self.encoder = nn.ModuleList([
            EncoderBlock(c_in=6, c_out=64, k=7, p=3, batch_norm=batch_norm, two_layer=False),
            EncoderBlock(c_in=64, c_out=128, k=5, p=2, batch_norm=batch_norm, two_layer=False),
            EncoderBlock(c_in=128, c_out=256, k=5, p=2, batch_norm=batch_norm),
            EncoderBlock(c_in=256, c_out=512, k=3, batch_norm=batch_norm),
            EncoderBlock(c_in=512, c_out=512, k=3, batch_norm=batch_norm),
            EncoderBlock(c_in=512, c_out=1024, k=3, batch_norm=batch_norm)
        ])

        self.predictor = nn.ModuleList([Predictor(c_in=1024 // 2 ** i) for i in range(6)])

        self.decoder = nn.ModuleList([
            DecoderBlock(c_in=1024, merger_c_in=1025),
            DecoderBlock(c_in=512),
            DecoderBlock(c_in=256),
            DecoderBlock(c_in=128),
            DecoderBlock(c_in=64)
        ])

    def forward(self, x):
        contracting_x = ()
        predictions = ()
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.verbose:
                print(f"encoder{i}->{x.shape}")
            contracting_x = contracting_x + (x,)

        pred = self.predictor[0](x)
        predictions = predictions + (pred,)
        if self.verbose:
            print(f"prediction6->{predictions[-1].shape}")

        for i, decoder, predictor in zip(range(5, 0, -1), self.decoder, self.predictor[1:]):
            x = decoder(x, predictions[-1], contracting_x[i - 1])
            if self.verbose:
                print(f"decoder{i}->{x.shape}")
            pred = predictor(x)
            predictions = predictions + (pred,)
            if self.verbose:
                print(f"prediction{i}->{predictions[-1].shape}")

        return predictions


class EncoderBlock(nn.Module):
    def __init__(self, c_in, c_out, k, p=1, two_layer=True, batch_norm=False):
        super(EncoderBlock, self).__init__()
        self.layer0 = EncoderConvLayer(c_in, c_out, k, stride=2, padding=p, batch_norm=batch_norm)
        self.layer1 = None
        if two_layer:
            self.layer1 = EncoderConvLayer(c_out, c_out, 3, batch_norm=batch_norm)

    def forward(self, x):
        x = self.layer0(x)
        if self.layer1:
            x = self.layer1(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, c_in, merger_c_in=None):
        super(DecoderBlock, self).__init__()
        if not merger_c_in:
            merger_c_in = c_in + c_in // 2 + 1

        self.refiner = prediction_upsampling_layer()
        self.decoder = DecoderConvLayer(c_in=c_in, c_out=c_in // 2)
        self.merger = nn.Conv2d(in_channels=merger_c_in, out_channels=c_in // 2, kernel_size=3, padding=1)

    def forward(self, x, prev_pred, prev_x):
        up_pred = self.refiner(prev_pred)
        x = self.decoder(x)
        x = self.merger(cat([x, up_pred, prev_x], 1))
        return x


class Predictor(nn.Module):
    def __init__(self, c_in):
        super(Predictor, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class EncoderConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, batch_norm=False):
        super(EncoderConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.batch_norm = None
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=c_out)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.relu(x)


class DecoderConvLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(DecoderConvLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


def prediction_upsampling_layer():
    return nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)


if __name__ == "__main__":
    from torch import randn

    original_res = (384, 768)
    network = NNModel(verbose=True)
    dummy_input = randn((1, 6, *original_res))
    output = network(dummy_input)
    # for j in network.named_parameters():
    #    print(j[0], j[1].size())
