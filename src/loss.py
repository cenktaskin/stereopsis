import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from dataset import StereopsisDataset, data_path


class MaskedMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y, i):
        if i == 0:
            print(f"{yhat.size()=}")
            print(f"{y.size()=}")
        yhat[y == 0] = 0  # mask the 0.0 elements to not to contribute to the error
        return torch.sqrt(self.mse(yhat, y))


if __name__ == "__main__":
    dataset_id = "20220610"
    dataset_path = data_path.joinpath(f"raw/dataset-{dataset_id}")
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    dataset = StereopsisDataset(dataset_path)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    x, y = next(iter(train_dataloader))
    pred = torch.randn((batch_size, 1, 90, 160))
    y = interpolate(y.unsqueeze(dim=1), size=pred.shape[-2:], mode="nearest-exact")

    loss_fn = MaskedMSE()
    loss = loss_fn(pred, y)
    print(loss.item())
