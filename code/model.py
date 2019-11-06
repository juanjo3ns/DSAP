import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from IPython import embed

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = None
        self.params = {}

        self.CNN_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7,7), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,5), stride=(1,1)),
            nn.Dropout(0.3))

        self.CNN_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(7,7), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,100), stride=(1,1)),
            nn.Dropout(0.3))

        self.FC = nn.Sequential(
            nn.Linear(124480, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Softmax(dim=1))


    def forward(self, xb):
        out = self.CNN_1(xb)
        out = self.CNN_2(out)
        out = out.view(out.shape[0], -1)
        out = self.FC(out)
        return out


class WAV_model_test(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = None
        self.params = {}
        self.params["gru_hidden_size"] = 1000

        self.CNN_1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=(5,1)))

        self.CNN_2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1), stride=(4,1)))

        self.CNN_3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)))

        self.gru = nn.GRU(input_size=6912, hidden_size=self.params["gru_hidden_size"], num_layers=1, batch_first=True, dropout=0.25)

        self.FC = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid())



    #nn.Softmax(dim=1)

    def forward(self, xb):

        self.hidden = torch.zeros([1, 1, self.params["gru_hidden_size"]], dtype=torch.float32).cuda()

        out = self.CNN_1(xb)
        out = self.CNN_2(out)
        out = self.CNN_3(out)

        out = out.view(1, out.shape[1]*out.shape[2], 1480).permute(0,2,1)
        out, self.hidden = self.gru(out, self.hidden)
        out = self.FC(out)
        return out

        """
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.layer3(out)
        #out = self.fc2(out)
        #out = nn.ReLU(out)
        return out
        """



class WAV_model_proba(nn.Module):
    def __init__(self):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(6440960, 10)
        self.layer3 =nn.Sequential(
            nn.Linear(10, 2),
            nn.PReLU(num_parameters=1, init=0.25)
            )
    #nn.Softmax(dim=1)

    def forward(self, xb):


        out = self.layer1(xb)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.layer3(out)
        #out = self.fc2(out)
        #out = nn.ReLU(out)
        return out

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.maxPool1 = nn.MaxPool2d(2)
        self.maxPool2 = nn.MaxPool2d(4)

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):

        #falta batch Normalitzation

        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxPool1(out)

        out = self.conv2(x)
        out = self.relu(out)
        out = self.maxPool1(out)

        out = self.conv3(x)
        out = self.relu(out)
        out = self.maxPool2(out)

        # Falta incloure la concatenacio de les dues branques (de moment nomes una)
        """
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        """
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
