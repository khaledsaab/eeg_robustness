from torch import nn


class Conv_1D(nn.Module):
    def __init__(self, d_input, d_model, d_output, kernel_size=200, forecasting=False):
        super().__init__()

        self.forecasting = forecasting
        self.d_output = d_output
        # self.encoder = nn.Sequential(nn.Linear(d_input, d_model), nn.GELU())
        self.encoder = nn.Linear(d_input, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=kernel_size)
        if self.forecasting:
            self.decoder = nn.Linear(d_model, d_output)  
        else:
            self.decoder = nn.Sequential(nn.Linear(d_model, d_output), nn.GELU())

    def forward(self, x):
        y = self.encoder(x)
        y = y.transpose(-1, -2)
        y = self.conv1d(y)
        y = y.transpose(-1, -2)
        y = self.decoder(y)
        #y = y.mean(-2) 
        y = y[:,-1,:]
        if self.forecasting:
            y = y.unsqueeze(-2)
        return y
