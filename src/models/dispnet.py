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

    def forward(self, x00):
        x1b = self.encoder[0](x00)
        x2b = self.encoder[1](x1b)
        x3a = self.encoder[2](x2b)
        x3b = self.encoder[3](x3a)
        x4a = self.encoder[4](x3b)
        x4b = self.encoder[5](x4a)
        x5a = self.encoder[6](x4b)
        x5b = self.encoder[7](x5a)
        x6a = self.encoder[8](x5b)
        x6b = self.encoder[9](x6a)

        pr6 = self.prediction6(x6b)

        block = self.decoder[0]
        ux5 = block["decoder"](x6b)
        ix5 = block["merger"](cat([ux5, block["refiner"](pr6), x5b], 1))
        pr5 = block["predictor"](ix5)

        block = self.decoder[1]
        ux4 = block["decoder"](ix5)
        ix4 = block["merger"](cat([ux4, block["refiner"](pr5), x4b], 1))
        pr4 = block["predictor"](ix4)

        block = self.decoder[2]
        ux3 = block["decoder"](ix4)
        ix3 = block["merger"](cat([ux3, block["refiner"](pr4), x3b], 1))
        pr3 = block["predictor"](ix3)

        block = self.decoder[3]
        ux2 = block["decoder"](ix3)
        ix2 = block["merger"](cat([ux2, block["refiner"](pr3), x2b], 1))
        pr2 = block["predictor"](ix2)

        block = self.decoder[4]
        ux1 = block["decoder"](ix2)
        ix1 = block["merger"](cat([ux1, self.decoder[4]["refiner"](pr2), x1b], 1))
        pr1 = block["predictor"](ix1)

        return pr6, pr5, pr4, pr3, pr2, pr1


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
