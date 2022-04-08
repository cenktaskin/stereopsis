from torch import nn, sqrt


class MaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        y = y.unsqueeze(dim=1)
        yhat[y == 0] = 0  # mask the 0.0 elements to not to contribute to the error
        return sqrt(self.mse(yhat, y))



#class MaskedMSE(nn.Module):
#    def __init__(self):
#        super().__init__()
#        #self.mse = nn.MSELoss()
#
#    def forward(self, yhat, y):
#        mask = y == 0
#        d = (yhat[~mask] - y[~mask]) ** 2  # mask the 0.0 elements to not to contribute to the error
#        loss = d.mean()
#        return loss
