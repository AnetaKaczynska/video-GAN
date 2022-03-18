import math

import torch
import torch.nn as nn
import torch.functional as F


class FrameSeedGenerator(nn.Module):
    def __init__(self):
        super(FrameSeedGenerator, self).__init__()
        self.D = 512
        self.fc1 = nn.Linear(2048, 1024)   # d = 2048
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)     # D = 512

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = torch.sin(self.fc3(x))
        # normalize
        x = x / torch.norm(x) * math.sqrt(self.D)
        return x
        