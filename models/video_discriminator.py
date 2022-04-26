import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoDiscriminator(nn.Module):
    def __init__(self, active=False):
        super(VideoDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(512, 256, 4, 1)
        self.conv2 = nn.Conv1d(256, 128, 3, 1)
        self.conv3 = nn.Conv1d(128, 64, 3, 1)
        self.fc = nn.Linear(64, 1)
        if active:
            self.activ_function = nn.Sigmoid()
        self.active = active

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.activ_function(x) if self.active else x


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    n_frames = 8
    h = 1024
    w = h
    video = torch.rand(n_frames, 512)   # (N, 512)
    video = video.permute(1, 0).unsqueeze(0)
    net = VideoDiscriminator() # .cuda()
    # x = torch.randn(1, 3, n_frames, h, w).cuda()  # (1, 3, 8, 1024, 1024)

    output = net(video)
    # from torchvision.utils import save_image
    # save_image(output, nrow=8, normalize=True, fp='test_img2.jpg')