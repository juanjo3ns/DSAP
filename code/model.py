import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from IPython import embed

class BaselineModel(nn.Module):
    def __init__(self, num_classes=10, task=5):
        super().__init__()
        self.num_classes = num_classes
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

        if task == 1:
            self.FC = nn.Sequential(
                nn.Linear(124480, 100),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(100, self.num_classes),
                nn.ReLU(),
                nn.Softmax(dim=1))
        elif task == 5:
            self.FC = nn.Sequential(
                nn.Linear(124480, 100),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(100, self.num_classes),
                nn.ReLU(),
                nn.Sigmoid())



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

        self.CNN_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            nn.ReLU())

        self.gru = nn.GRU(input_size=385, hidden_size=self.params["gru_hidden_size"], num_layers=1, batch_first=True, dropout=0.25)

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
        embed()

        out = out.view(128, out.shape[1]*out.shape[2], 385)
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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=8, p_dropout=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc1 = nn.Linear(8192, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout(p=p_dropout)
        self.sigm = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)    # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)   # 56x56
        x = self.layer2(x)   # 28x28
        x = self.layer3(x)   # 14x14
        x = self.layer4(x)   # 7x7

        x = self.avgpool(x)  # 1x1
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        x = self.sigm(x)

        return x

def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    num_param = sum(p.numel() for p in model.parameters())
    return model, num_param/1000000

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    num_param = sum(p.numel() for p in model.parameters())
    return model, num_param/1000000

def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    num_param = sum(p.numel() for p in model.parameters())
    return model, num_param/1000000



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
