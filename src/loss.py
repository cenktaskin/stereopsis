import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
import time
from dataset import StereopsisDataset, data_path


class MaskedMSE(torch.nn.Module):
    # rows are for each loss layer from 6 to 1
    loss_schedule = torch.tensor([[1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                  [0.5, 1.0, 0.2, 0.0, 0.0, 0.0],
                                  [0.0, 0.5, 1.0, 0.2, 0.0, 0.0],
                                  [0.0, 0.0, 0.5, 1.0, 0.2, 0.0],
                                  [0.0, 0.0, 0.0, 0.5, 1.0, 0.2],
                                  [0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        y = y.unsqueeze(dim=1)
        if yhat.shape[-2:] != y.shape[-2:]:
            yhat = interpolate(yhat, size=y.shape[-2:], mode="bilinear")
        yhat[y == 0] = 0  # mask the 0.0 elements
        return torch.sqrt(self.mse(yhat, y))


if __name__ == "__main__":
    dataset_id = "20220610"
    dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}")
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    dataset = StereopsisDataset(dataset_path)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    x, y = next(iter(train_dataloader))
    pred = torch.randn((batch_size, 1, 24, 48))

    initial_size = torch.tensor([6, 12])
    preds = tuple([torch.randn(batch_size, 1, *initial_size * 2 ** i) for i in range(6)])

    if y.dim() == 3:  # for grayscale img
        y = y.unsqueeze(dim=1)
    current_breakpoint = 0
    loss = 0.0
    current_loss_weights = MaskedMSE.loss_schedule[current_breakpoint]
    current_loss_weights = current_loss_weights / current_loss_weights.sum()

    t0 = time.time()
    for i in range(100):
        for i in current_loss_weights.nonzero():
            upsampled_pred = interpolate(preds[i], size=y.shape[-2:], mode="bilinear")
            curr_loss = torch.sqrt(torch.nn.functional.mse_loss(upsampled_pred, y)) * current_loss_weights[i]
            loss += curr_loss
    print(f"Took {time.time() - t0} secs")

    t0 = time.time()
    for i in range(100):
        loss_volume = torch.zeros(*y.shape, len(preds))
        for i, w in enumerate(current_loss_weights):
            if w > 0:
                loss_volume[..., i] = interpolate(preds[i], size=y.shape[-2:], mode="bilinear")
        weighted_loss_volume = loss_volume @ current_loss_weights
    print(f"Took {time.time() - t0} secs")
