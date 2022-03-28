import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoDiscriminator(nn.Module):

    def __init__(self, in_channels, top_width, mid_ch):
        super(VideoDiscriminator, self).__init__()
        self.max_pool = nn.MaxPool3d((1, 16, 16))   # to make input shape be (1, 3, 8, 64, 64)
        k = 3
        self.c1 = nn.Conv3d(in_channels, mid_ch, k, 2, 1)
        self.c2 = nn.Conv3d(mid_ch, mid_ch * 2, k, 2, 1)
        self.c3 = nn.Conv3d(mid_ch * 2, mid_ch * 4, k, 2, 1)
        self.c4 = nn.Conv3d(mid_ch * 4, mid_ch * 8, k, 2, 1)
        self.c5 = nn.Conv2d(mid_ch * 8, 1, top_width, 1, 0)
        self.bn1 = nn.BatchNorm3d(mid_ch * 2)
        self.bn2 = nn.BatchNorm3d(mid_ch * 4)
        self.bn3 = nn.BatchNorm3d(mid_ch * 8)

    def __call__(self, x):
        # (N, CH, T, H, W)
        assert x.shape[2:] == (8, 1024, 1024), "works only for 8 frames and 1024x1024 resolution for now"

        x = self.max_pool(x)
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.bn1(self.c2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.c3(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.c4(x)), 0.2)
        # x = x.reshape((x.shape[0] * x.shape[2],) + self.c4.weight.shape[1:])
        x = x.squeeze(2)
        x = self.c5(x)
        return x

if __name__ == '__main__':
    n_frames = 8
    h = 1024
    w = h
    net = VideoDiscriminator(3, 4, 64).cuda()
    x = torch.randn(1, 3, n_frames, h, w).cuda()
    print(net(x).shape)