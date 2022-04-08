from torch import nn, sqrt


class MaskedRMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        yhat[y == 0] = 0  # mask the 0.0 elements to not to contribute to the error
        return sqrt(self.mse(yhat, y))
