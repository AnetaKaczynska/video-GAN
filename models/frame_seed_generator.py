import math

import torch
import torch.nn as nn
import torch.linalg as LA
import torch.functional as F


class FrameSeedGenerator(nn.Module):

    def __init__(self):
        super(FrameSeedGenerator, self).__init__()
        self.D = 512
        # 1 wariant
        # self.fc1 = nn.Linear(2048, 2559)   # d = 2048
        # self.fc2 = nn.Linear(2560, 1023)
        # self.fc3 = nn.Linear(1024, 511)
        # self.fc4 = nn.Linear(512, 512)     # D = 512

        # 2 wariant
        self.fc1 = nn.Linear(2048, 1535)   # d = 2048
        self.fc2 = nn.Linear(1536, 1023)
        self.fc3 = nn.Linear(1024, 511)
        self.fc4 = nn.Linear(512, 512)     # D = 512

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, seeds, time):
        x = torch.hstack([time, seeds])
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.hstack([time, x])
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = torch.hstack([time, x])
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = torch.hstack([time, x])
        x = self.fc4(x)
        # normalize
        x = x / LA.norm(x, dim=1, keepdim=True) * math.sqrt(self.D)
        return x
