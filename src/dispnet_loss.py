from torch import nn
from torch.nn.functional import interpolate
import torch
from torch.utils.data import DataLoader

from dataset import StereopsisDataset, data_path


class MaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        #b, h, w = y.shape
        y = y.unsqueeze(dim=1)
        #if (h, w) != yhat.shape[-2:]:
        #    print("needs interpolating")
        #    yhat = interpolate(yhat, size=(h, w), mode="area")
        #else:
        #    print("doesn't need")
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

    pred = torch.randn((batch_size, 1, 90, 160)) * 0.1 + interpolate(y.unsqueeze(dim=1), (90, 160), mode="area")
    loss_fn = MaskedMSE()
    loss = loss_fn(pred, y)
    print(loss.item())
