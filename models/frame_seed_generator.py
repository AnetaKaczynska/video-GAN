import math

import torch
import torch.nn as nn
import torch.linalg as LA


class FrameSeedGenerator(nn.Module):
    def __init__(self):
        super(FrameSeedGenerator, self).__init__()
        self.D = 512
        self.fc1 = nn.Linear(2048 + self.D, 1023)
        self.fc2 = nn.Linear(1024 + self.D, 511)
        self.fc3 = nn.Linear(512 + self.D, 512)

    def forward(self, seeds, noise, time):
        one_hot_len = noise.shape[1] - self.D
        bs = noise.shape[0]
        duration = time.shape[0] // bs
        pro_noise = noise[:, :-one_hot_len].tile(duration, 1)

        x = torch.hstack([pro_noise, time, seeds])
        x = torch.relu(self.fc1(x))
        x = torch.hstack([pro_noise, time, x])
        x = torch.relu(self.fc2(x))
        x = torch.hstack([pro_noise, time, x])
        x = self.fc3(x)

        x = pro_noise + x - x[:, None, 0]

        # normalize
        x = x / LA.norm(x, dim=1, keepdim=True) * math.sqrt(self.D)
        # append one hot for generator
        one_hots = noise[:, -one_hot_len:]
        x = torch.cat([x, one_hots.repeat(x.shape[0] // one_hots.shape[0], 1)], dim=1)
        return x