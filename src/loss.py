from torch import nn, sqrt, randn


class MaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        y = y.unsqueeze(dim=1)
        yhat[y == 0] = 0  # mask the 0.0 elements to not to contribute to the error
        return sqrt(self.mse(yhat, y))


if __name__ == "__main__":

    yhat = randn(1, 1, 2, 3)
    y = randn(1, 2, 3)
    y[0,1,1] = 0
    diff  = yhat-y
    print(yhat)
    print(y)
    loss_fn = MaskedMSE()
    loss = loss_fn(yhat, y)
    print(loss.item())