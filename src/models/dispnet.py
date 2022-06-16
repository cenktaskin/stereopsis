from torch import nn, cat
from torch.nn.functional import leaky_relu


class NNModel(nn.Module):
    name = 'dispnet'

    def __init__(self, batch_norm=False):
        super(NNModel, self).__init__()

        self.encoder = nn.ModuleList([
            encoder_block(c_in=6, c_out=64, batch_norm=batch_norm, kernel_size=7, stride=2, padding=3),
            encoder_block(c_in=64, c_out=128, batch_norm=batch_norm, kernel_size=5, stride=2, padding=2),
            encoder_block(c_in=128, c_out=256, batch_norm=batch_norm, kernel_size=5, stride=2, padding=2),
            encoder_block(c_in=256, c_out=256, batch_norm=batch_norm, kernel_size=3),
            encoder_block(c_in=256, c_out=512, batch_norm=batch_norm, kernel_size=3, stride=2),
            encoder_block(c_in=512, c_out=512, batch_norm=batch_norm, kernel_size=3),
            encoder_block(c_in=512, c_out=512, batch_norm=batch_norm, kernel_size=3, stride=2),
            encoder_block(c_in=512, c_out=512, batch_norm=batch_norm, kernel_size=3),
            encoder_block(c_in=512, c_out=1024, batch_norm=batch_norm, kernel_size=3, stride=2),
            encoder_block(c_in=1024, c_out=1024, batch_norm=batch_norm, kernel_size=3),
        ])

        self.prediction6 = prediction_layer(c_in=1024)

        self.decoder = nn.ModuleList([
            decoder_block(c_in=1024, iconv_c_in=1025),
            decoder_block(c_in=512),
            decoder_block(c_in=256),
            decoder_block(c_in=128),
            decoder_block(c_in=64)
        ])

    def forward(self, x):
        contracting_x = {}
        predictions = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 0 or i % 2 == 1:
                contracting_x[len(contracting_x) + 1] = x

        predictions.append(self.prediction6(x))

        for i, block in zip(range(5, 0, -1), self.decoder):
            x = block["decoder"](x)
            x = block["merger"](cat([x, block["refiner"](predictions[-1]), contracting_x[i]], 1))
            predictions.append(block["predictor"](x))

        return predictions


def encoder_block(c_in, c_out, batch_norm, padding=1, **kwargs):
    modules = [nn.Conv2d(in_channels=c_in, out_channels=c_out, padding=padding, **kwargs)]
    if batch_norm:
        modules.append(nn.BatchNorm2d(num_features=c_out))
    modules.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    return nn.Sequential(*modules)


def decoder_block(c_in, iconv_c_in=None):
    if not iconv_c_in:
        iconv_c_in = c_in + c_in // 2 + 1
    return nn.ModuleDict({
        "refiner": prediction_upsampling_layer(),
        "decoder": decoder_conv_layer(c_in=c_in, c_out=c_in // 2),
        "merger": nn.Conv2d(in_channels=iconv_c_in, out_channels=c_in // 2, kernel_size=3, padding=1),
        "predictor": prediction_layer(c_in // 2)
    })


def decoder_conv_layer(c_in, c_out):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


def prediction_layer(c_in):
    return nn.Conv2d(in_channels=c_in, out_channels=1, kernel_size=3, stride=1, padding=1)


def prediction_upsampling_layer():
    return nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)


if __name__ == "__main__":
    from torch import randn

    original_res = (384, 768)
    network = NNModel()
    dummy_input = randn((1, 6, *original_res))
    output = network(dummy_input)
    for j in network.named_parameters():
        print(j[0], j[1].size())
