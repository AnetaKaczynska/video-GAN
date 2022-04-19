import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoDiscriminator(nn.Module):
    def __init__(self):
        super(VideoDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(512, 256, 3, 1)
        self.conv2 = nn.Conv1d(256, 128, 3, 1)
        self.conv3 = nn.Conv1d(128, 64, 3, 1)
        self.fc = nn.Linear(128, 1)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
    net = VideoDiscriminator() # .cuda()
    # x = torch.randn(1, 3, n_frames, h, w).cuda()  # (1, 3, 8, 1024, 1024)

    img = torch.ones(1, 8, 512).permute(0,2,1)
    print(img.shape)
    output = net(img)
    # from torchvision.utils import save_image
    # save_image(output, nrow=8, normalize=True, fp='test_img2.jpg')