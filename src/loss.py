import torch
from torch.nn.functional import interpolate


class MultilayerSmoothL1(torch.nn.Module):
    name = "MultilayerSmoothL1"
    # cols are for each loss layer from 6 to 0
    # last line is added by me
    old_weights = torch.tensor([[1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                                [0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
                                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=False)

    weights = torch.tensor([[0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
                            [0.005, 0.01, 0.02, 0.04, 0.08, 0.32, 0.6],
                            [0.0025, 0.005, 0.01, 0.02, 0.04, 0.16, 0.8],
                            [0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)

    def __init__(self):
        super().__init__()
        self.smoothl1 = torch.nn.SmoothL1Loss()

    def forward(self, predictions, label, stage):
        if stage > 3:
            stage = 3
        label = assert_label_dims(label)
        loss = 0
        for i in self.weights[stage].nonzero():
            upsampled_prediction = interpolate(predictions[i], size=label.shape[-2:], mode="bilinear")
            valid_pixels = label > 0
            valid_pixels.detach_()
            loss += self.smoothl1(upsampled_prediction[valid_pixels], label[valid_pixels]) * self.weights[stage][
                i].item()

        return loss


class MultilayerSmoothL1viaPool(torch.nn.Module):
    name = "MultilayerSmoothL1viaPool"
    # cols are for each loss layer from 6 to 0
    # last line is added by me
    weights = torch.tensor([[1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                            [0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
                            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=False)

    def __init__(self):
        super().__init__()
        self.smoothl1 = torch.nn.SmoothL1Loss()
        self.multiScales = [torch.nn.AvgPool2d(kernel_size=2 ** i) for i in range(6, 0, -1)]

    def forward(self, predictions, label, stage):
        if stage > 3:
            stage = 3
        label = assert_label_dims(label)
        loss = 0
        for i in self.weights[stage].nonzero():
            downsampled_label = self.multiScales[i](label)
            loss += self.smoothl1(predictions[i], downsampled_label) * self.weights[stage][i].item()
        return loss


class MaskedEPE(torch.nn.Module):
    name = "MaskedEPE"

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def rmse(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

    def forward(self, predictions, label):
        full_res_pred = predictions[-1].detach()
        y = assert_label_dims(label).detach()
        upsampled_pred = interpolate(full_res_pred, size=y.shape[-2:], mode="bilinear")
        valid_pix = y > 0
        return self.rmse(upsampled_pred[valid_pix], y[valid_pix])


def assert_label_dims(y):
    if y.dim() == 3:  # for grayscale img
        y = y.unsqueeze(dim=1)
    return y


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import StereopsisDataset, data_path

    dataset_id = "20220610-origres"
    dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}")
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    dataset = StereopsisDataset(dataset_path)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    sample_x, sample_y = next(iter(train_dataloader))

    initial_size = torch.tensor([6, 12])
    pr_list = tuple([torch.randn(batch_size, 1, *initial_size * 2 ** i) for i in range(6)])

    loss_fn = MaskedEPE()
    print(loss_fn.name)
    print(loss_fn(pr_list, sample_y, 1))
