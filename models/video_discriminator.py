import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


class VideoDiscriminator(nn.Module):

    def __init__(self, in_channels, top_width, mid_ch):
        super(VideoDiscriminator, self).__init__()
        k = 3
        self.net = nn.Sequential(
            nn.AvgPool3d((1, 8, 8), (1, 8, 8)),
            nn.Conv3d(in_channels, mid_ch//2, k, (1, 2, 2), 1),   # (1, 32, 8, 64, 64)
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(mid_ch//2, mid_ch, k, (1, 2, 2), 1),        # (1, 64, 8, 32, 32)
            nn.BatchNorm3d(mid_ch), 
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(mid_ch, mid_ch * 2, k, 2, 1),               # (1, 128, 4, 16, 16)
            nn.BatchNorm3d(mid_ch * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(mid_ch * 2, mid_ch * 4, k, 2, 1),           # (1, 256, 2, 8, 8)
            nn.BatchNorm3d(mid_ch * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(mid_ch * 4, mid_ch * 8, k, 2, 1),           # (1, 512, 1, 4, 4)
            nn.BatchNorm3d(mid_ch * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.c5 = nn.Conv2d(mid_ch * 8, 1, top_width, 1, 0)

    def __call__(self, x):
        # (N, CH, T, H, W)
        assert x.shape[2:] == (8, 1024, 1024), "works only for 8 frames and 1024x1024 resolution for now"

        x = self.net(x)
        x = x.squeeze(2)
        x = self.c5(x)
        return x


if __name__ == '__main__':
    import sys

    sys.path.append('../')
    from dataset.real_videos import RealVideos
    from torch.utils.data import DataLoader
    real_videos = RealVideos()
    dataloader = DataLoader(real_videos, batch_size=1, shuffle=True)
    img = next(iter(dataloader)) # .cuda()

    n_frames = 8
    h = 1024
    w = h
    net = VideoDiscriminator(3, 4, 64) # .cuda()
    # x = torch.randn(1, 3, n_frames, h, w).cuda()  # (1, 3, 8, 1024, 1024)

    print(net)
    # output = net(img).squeeze().permute([1, 0, 2, 3])
    # from torchvision.utils import save_image
    # save_image(output, nrow=8, normalize=True, fp='test_img2.jpg')