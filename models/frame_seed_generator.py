import math

import torch
import torch.nn as nn
import torch.linalg as LA


class FrameSeedGenerator(nn.Module):
    def __init__(self):
        super(FrameSeedGenerator, self).__init__()
        self.D = 512
        self.fc1 = nn.Linear(2048, 1023)   # d = 2048
        self.fc2 = nn.Linear(1024, 511)
        self.fc3 = nn.Linear(512, 512)     # D = 512

    def forward(self, seeds, noise, time):
        x = torch.hstack([time, seeds])
        x = torch.relu(self.fc1(x))
        x = torch.hstack([time, x])
        x = torch.relu(self.fc2(x))
        x = torch.hstack([time, x])
        x = self.fc3(x)

        one_hot_len = noise.shape[1] - x.shape[1]
        bs = noise.shape[0]
        # x = x.reshape(bs, -1, 512)
        # # we want to start from noise
        # x = noise[:, :-one_hot_len].unsqueeze(1) + x - x[:, None, 0]
        # x = x.reshape(-1, 512)

        # normalize
        # x = x / LA.norm(x, dim=1, keepdim=True) * math.sqrt(self.D)
        # append one hot for generator
        one_hots = noise[:, -one_hot_len:]
        x = torch.cat([x, one_hots.repeat(x.shape[0] // one_hots.shape[0], 1)], dim=1)
        return x