import os
import re
import glob
from itertools import groupby
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RealVideos(Dataset):
    def __init__(self):
        self.img_dir = '/shared/results/Skopia/images2'
        self.ch, self.duration, self.h, self.w = 3, 8, 1024, 1024
        self.videos = self._load_videos()
        self.transform=transforms.ToTensor()

    def _load_videos(self):
        videos = []
        images = os.listdir(self.img_dir)
        images = sorted(images)

        frames_per_video = groupby(images, key=lambda x: x[:5])
        for video, frames in frames_per_video:
            frames = list(frames)
            n = len(frames) % self.duration
            del frames[-n:]
            videos += [frames[n: n+self.duration] for n in range(0, len(frames), self.duration)]

        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        video = np.empty((self.ch, self.duration, self.h, self.w), dtype=np.float32)
        ts = 0
        frames = self.videos[i]
        for t, filename in enumerate(frames):
            filepath = self.img_dir + '/' + filename
            img = np.asarray(Image.open(filepath).resize((self.h, self.w)), dtype=np.uint8).transpose(2, 0, 1)
            video[:, t] = img

        return torch.as_tensor(video)

if __name__ == '__main__':
    real_videos = RealVideos()
    print(len(real_videos))
    for i, item in enumerate(real_videos):
        print(i)
        break
