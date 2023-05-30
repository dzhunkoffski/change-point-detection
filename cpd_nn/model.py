from torch import nn
import torch
import numpy as np

class CPD_LSTM(nn.Module):
    def __init__(self, lstm_params: dict, linear_params: dict):
        super(CPD_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size = lstm_params['input_dim'], 
                            hidden_size = lstm_params['hidden_dim'],
                            num_layers = lstm_params['num_layers'], 
                            dropout = lstm_params['dropout'], 
                            bidirectional = lstm_params['is_bidirectional'],
                            batch_first = True)

        hidden_dim = lstm_params['hidden_dim']
        if lstm_params['is_bidirectional']:
            hidden_dim *= 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        x, (h_n, _) = self.lstm(x)
        x = self.classifier(x[:,-1,:])
        return x

class CPD_RNN(nn.Module):
    def __init__(self, lstm_params: dict, linear_params: dict):
        super(CPD_RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size = lstm_params['input_dim'],
            hidden_size = lstm_params['hidden_dim'],
            num_layers = lstm_params['num_layers'],
            nonlinearity = lstm_params['nonlinearity'],
            batch_first = True,
            bidirectional = lstm_params['is_bidirectional']
        )
        hidden_dim = lstm_params['hidden_dim']
        if lstm_params['is_bidirectional']:
            hidden_dim *= 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.classifier(x[:,-1,:])
        return x

class CPD_GRU(nn.Module):
    def __init__(self, lstm_params: dict, linear_params: dict):
        super(CPD_GRU, self).__init__()

        self.gru = nn.GRU(
            input_size = lstm_params['input_dim'],
            hidden_size = lstm_params['hidden_dim'],
            num_layers = lstm_params['num_layers'],
            batch_first = True,
            dropout = lstm_params['dropout']
        )

        hidden_dim = lstm_params['hidden_dim']
        if lstm_params['is_bidirectional']:
            hidden_dim *= 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.classifier(x[:, -1, :])
        return x


    
class CPD_MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CPD_MLP, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x

class CPD_LSTM2(nn.Module):
    def __init__(self, lstm_params: dict, linear_params: dict):
        super(CPD_LSTM2, self).__init__()

        self.lstm = nn.LSTM(input_size = lstm_params['input_dim'], 
                            hidden_size = lstm_params['hidden_dim'],
                            num_layers = lstm_params['num_layers'], 
                            dropout = lstm_params['dropout'], 
                            bidirectional = lstm_params['is_bidirectional'],
                            batch_first = True)
        hidden_dim = lstm_params['hidden_dim']
        if lstm_params['is_bidirectional']:
            hidden_dim *= 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, linear_params['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(linear_params['hidden_dim']),
            nn.Linear(linear_params['hidden_dim'], linear_params['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(linear_params['hidden_dim']),
            nn.Linear(linear_params['hidden_dim'], linear_params['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(linear_params['hidden_dim']),
            nn.Linear(linear_params['hidden_dim'], linear_params['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(linear_params['hidden_dim']),
            nn.Linear(linear_params['hidden_dim'], 1),
        )
    
    def forward(self, x):
        x, (h_n, _) = self.lstm(x)
        x = self.classifier(x[:,-1,:])
        return x