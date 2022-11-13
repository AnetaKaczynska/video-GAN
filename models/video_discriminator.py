import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class VideoDiscriminator(nn.Module):
    def __init__(self, active=False):
        super(VideoDiscriminator, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Conv2d(1, 16, (5, 10), (1, 2)),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 12, (3, 8), (1, 2)),
        #     nn.BatchNorm2d(12),
        #     nn.ReLU(True),
        #     nn.Conv2d(12, 8, (5, 6), (1, 2)),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 4, (3, 6), (1, 2)),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(True),
        #     nn.Conv2d(4, 2, (5, 6), (1, 2)),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(True),
        #     nn.Conv2d(2, 1, (3, 3), (1, 2)),
        #     nn.ReLU(True),
        # )
        self.net1 = nn.Sequential(
            nn.Conv2d(2, 16, (5, 10), (1, 2)),
            nn.ReLU(True),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(17, 12, (3, 8), (1, 2)),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(13, 8, (5, 6), (1, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(9, 4, (3, 6), (1, 2)),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
        )
        self.net5 = nn.Sequential(
            nn.Conv2d(5, 2, (5, 6), (1, 2)),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
        )
        self.net6 = nn.Sequential(
            nn.Conv2d(3, 1, (3, 3), (1, 2)),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(30, 1)
        if active:
            self.activ_function = nn.Sigmoid()
        self.active = active

    def __call__(self, x, time):
        x = x.unsqueeze(1)
        # x = self.net(x)

        align_corners = False
        # time: [batch_size, num_frames]
        time = time.unsqueeze(1)

        time_temp = time.unsqueeze(-1).tile(1, 1, 1, x.size(-1))
        x = torch.cat((time_temp, x), dim=1)
        x = self.net1(x)

        time_temp = interpolate(time, size=x.size(2), mode='linear', align_corners=align_corners)
        time_temp = time_temp.unsqueeze(-1).tile(1, 1, 1, x.size(-1))
        x = torch.cat((time_temp, x), dim=1)
        x = self.net2(x)

        time_temp = interpolate(time, size=x.size(2), mode='linear', align_corners=align_corners)
        time_temp = time_temp.unsqueeze(-1).tile(1, 1, 1, x.size(-1))
        x = torch.cat((time_temp, x), dim=1)
        x = self.net3(x)

        time_temp = interpolate(time, size=x.size(2), mode='linear', align_corners=align_corners)
        time_temp = time_temp.unsqueeze(-1).tile(1, 1, 1, x.size(-1))
        x = torch.cat((time_temp, x), dim=1)
        x = self.net4(x)

        time_temp = interpolate(time, size=x.size(2), mode='linear', align_corners=align_corners)
        time_temp = time_temp.unsqueeze(-1).tile(1, 1, 1, x.size(-1))
        x = torch.cat((time_temp, x), dim=1)
        x = self.net5(x)

        time_temp = interpolate(time, size=x.size(2), mode='linear', align_corners=align_corners)
        time_temp = time_temp.unsqueeze(-1).tile(1, 1, 1, x.size(-1))
        x = torch.cat((time_temp, x), dim=1)
        x = self.net6(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.activ_function(x) if self.active else x


if __name__ == '__main__':
    n_frames = 8
    h = 1024
    w = h
    video = torch.rand(n_frames, 512).unsqueeze(0)  # (1, N, 512)
    # video = video.permute(1, 0).unsqueeze(0)
    net = VideoDiscriminator()  # .cuda()
    # x = torch.randn(1, 3, n_frames, h, w).cuda()  # (1, 3, 8, 1024, 1024)

    output = net(video)
    # from torchvision.utils import save_image
    # save_image(output, nrow=8, normalize=True, fp='test_img2.jpg')