import torch
import torch.nn as nn


class VideoDiscriminator(nn.Module):
    def __init__(self, active=False, num_frame=24):
        super(VideoDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (3, 10), (1, 2)),
            nn.ReLU(True),
            nn.Conv2d(16, 8, (3, 8), (1, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 4, (3, 6), (1, 2)),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 1, (2, 6), (1, 2)),
            nn.ReLU(True),
        )
        if num_frame == 8:
            self.fc = nn.Linear(27, 1)
        elif num_frame == 24:
            self.fc = nn.Linear(459, 1)  # before 27
        else:
            raise NotImplemented()
        if active:
            self.activ_function = nn.Sigmoid()
        self.active = active

    def __call__(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
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
