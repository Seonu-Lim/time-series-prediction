import torch
from torch import nn, optim
from blitz.modules import BayesianLSTM

class LSTM(nn.Module) :
    def __init__(self, n_features, n_hidden,seq_len, y_length, n_layers=2):
        super(LSTM,self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            dropout = 0
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=y_length)


    def forward(self, sequences):
        lstm_out, self_hidden = self.lstm(sequences)
        last_time_step = lstm_out[:, -1]
        y_pred = self.linear(last_time_step)
        return y_pred


class B_LSTM(nn.Module) :
    def __init__(self, n_features, n_hidden,seq_len, y_length):
        super(B_LSTM,self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.y_len = y_length

        self.lstm = BayesianLSTM(n_features,n_hidden)
        self.linear = nn.Linear(in_features=n_hidden, out_features=y_length)

    def forward(self, sequences):
        lstm_out, self_hidden = self.lstm(sequences)
        last_time_step = lstm_out[:, -1,:]
        y_pred = self.linear(last_time_step)
        return y_pred
