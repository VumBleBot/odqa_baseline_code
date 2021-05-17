import random

import numpy as np
import torch
from torch import nn

# - embedding dim 768 기준입니다 / 추후 argument로 줄 수 있도록 수정 필요 

class LstmQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
        self.pooler = nn.Linear(1536, 2) # nn.AdaptiveAvgPool1d(-1)

    def forward(self, x):
        x, (_, _) = self.lstm(x) 
        x = self.pooler(x)

        return x


class CnnQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=2, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size, out_channels=2, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size, out_channels=2, kernel_size=5, padding=2)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1)
        conv3_out = self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1)
        conv5_out = self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1)
        x = conv1_out + conv3_out + conv5_out

        return x


class FcQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        x = self.fc(x)

        return x


class ComplexCnnQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1)
        conv3_out = self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1)
        conv5_out = self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1)
        output = self.fc(torch.cat((conv1_out, conv3_out, conv5_out), -1))

        return output