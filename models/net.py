# Importing torch libraries
import torch.nn as nn


class StumbleNet(nn.Module):

    def __init__(self, n_features, hidden_size, n_layers, dropout):
        super(StumbleNet, self).__init__()
        self.lstm = nn.LSTM(
            int(n_features),
            int(hidden_size),
            int(n_layers),
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    model = StumbleNet(337, 256, 2, 0.3)
    print(model)
