# Importing torch libraries
import torch.nn as nn
import torch


class StumbleNet(nn.Module):

    def __init__(self, n_features, hidden_size, n_layers, dropout, device):
        super(StumbleNet, self).__init__()

        self.ip_dim = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.lstm = nn.LSTM(
            input_size=int(n_features),
            hidden_size=int(hidden_size),
            num_layers=int(n_layers),
            dropout=dropout,
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hiddenState(self, batchsize):
        return torch.zeros(self.n_layers, batchsize, self.hidden_size)

    def forward(self, x):
        batchsize = x.shape[1]
        hidden_state = self.init_hiddenState(batchsize).to(self.device)
        cell_state = hidden_state
        out, _ = self.lstm(x, (hidden_state, cell_state))
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    model = StumbleNet(337, 256, 2, 0.3, 'cpu')
    print(model)
