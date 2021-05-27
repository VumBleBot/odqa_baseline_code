import torch
from torch import nn


class DprQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        x = self.fc(x)

        return x


class LstmQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.pooler = nn.Linear(input_size * 2, 2)

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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        x = conv1_out + conv3_out + conv5_out

        return x


class ComplexCnnQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        output = self.fc(torch.cat((conv1_out, conv3_out, conv5_out), -1))

        return output


class ComplexCnnQAHead_v2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        output = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1)))

        return output


class CnnLstmQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.conv_1 = nn.Conv1d(in_channels=input_size * 2, out_channels=256, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size * 2, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size * 2, out_channels=256, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768, 2)

    def forward(self, inputs):
        x, (_, _) = self.lstm(inputs)
        x = x.transpose(1, 2).contiguous()

        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        x = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1)))

        return x


class ComplexCnnEmQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.em_embedding = nn.Embedding(num_embeddings=384, embedding_dim=768)
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768, 2)

    def forward(self, inputs):
        """
        x (8, 384)
        exact_match_pos (8, 384)
            = [[0, 1, 0, 0, ..., 1, 0, ...], ...]
        """
        x, exact_match_pos = inputs
        x = x + self.em_embedding(exact_match_pos)

        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        output = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1)))

        return output


class ComplexCnnLstmEmQAHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.em_embedding = nn.Embedding(num_embeddings=384, embedding_dim=768)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.conv_1 = nn.Conv1d(in_channels=input_size * 2, out_channels=256, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size * 2, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size * 2, out_channels=256, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768, 2)

    def forward(self, inputs):
        """
        x (8, 384)
        exact_match_pos (8, 384)
            = [[0, 1, 0, 0, ..., 1, 0, ...], ...]
        """
        x, exact_match_pos = inputs
        x = x + self.em_embedding(exact_match_pos)

        x, (_, _) = self.lstm(x)
        x = x.transpose(1, 2).contiguous()

        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        output = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1)))

        return output
