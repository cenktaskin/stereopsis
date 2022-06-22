import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from dataset import StereopsisDataset, data_path


class MultilayerSmoothL1(torch.nn.Module):
    name = "MultilayerSmoothL1"
    # rows are for each loss layer from 6 to 1
    loss_schedule = torch.tensor([[1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                  [0.5, 1.0, 0.2, 0.0, 0.0, 0.0],
                                  [0.0, 0.5, 1.0, 0.2, 0.0, 0.0],
                                  [0.0, 0.0, 0.5, 1.0, 0.2, 0.0],
                                  [0.0, 0.0, 0.0, 0.5, 1.0, 0.2],
                                  [0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) # last line is added by me


    def __init__(self):
        super().__init__()
        self.smoothl1 = torch.nn.SmoothL1Loss()

    def forward(self, preds, y, round):
        y = assert_label_dims(y)

        current_weights = self.loss_schedule[round]

        loss = 0.0
        for i in current_weights.nonzero():
            upsampled_pred = interpolate(preds[i], size=y.shape[-2:], mode="bilinear")
            upsampled_pred[y == 0] = 0  # mask the 0.0 elements
            loss += self.smoothl1(upsampled_pred, y) * current_weights[i].item()

        return loss


class MaskedEPE(torch.nn.Module):
    name = "MaskedEPE"

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, preds, y):
        y = assert_label_dims(y)
        full_res_pred = preds[-1]
        upsampled_pred = interpolate(full_res_pred, size=y.shape[-2:], mode="bilinear")
        upsampled_pred[y == 0] = 0  # mask the 0.0 elements
        return torch.sqrt(self.mse(upsampled_pred, y))


def assert_label_dims(y):
    if y.dim() == 3:  # for grayscale img
        y = y.unsqueeze(dim=1)
    return y


if __name__ == "__main__":
    dataset_id = "20220610-origres"
    dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}")
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    dataset = StereopsisDataset(dataset_path)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    sample_x, sample_y = next(iter(train_dataloader))

    initial_size = torch.tensor([6, 12])
    pr_list = tuple([torch.randn(batch_size, 1, *initial_size * 2 ** i) for i in range(6)])

    loss_fn = MaskedMSE()
    print(loss_fn.name)
    print(loss_fn(pr_list, sample_y, 1))
