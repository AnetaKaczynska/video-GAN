import os
import re

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RealVideos(Dataset):
    def __init__(self):
        self.img_dir = '/shared/results/Skopia/videos_24frames'
        self.ch, self.duration, self.h, self.w = 3, 24, 16, 16
        self.frames = 12
        self.video_names = self._load_videos()

    def _load_videos(self):
        images = os.listdir(self.img_dir)
        img_name = re.compile('([0-9]+_[0-9]+)_[0-9]+.jpg')
        video_names = {img_name.search(img).groups()[0] for img in images if img.endswith('_24.jpg')}
        return list(sorted(video_names))

    def __len__(self):
        return len(self.video_names) * 2   # each video contains 2*12 frames

    def __getitem__(self, i):
        """Return images for 12 consecutive frames."""
        video = np.empty((self.frames, self.ch, self.h, self.w), dtype=np.float32)
        video_name = self.video_names[i//2]
        m = i % 2

        for frame in range(self.frames*m, self.frames+self.frames*m):
            img_path = self.img_dir + f'/{video_name}_{frame+1}.jpg'
            img = np.asarray(Image.open(img_path).resize((self.h, self.w), Image.LANCZOS), dtype=np.float32).transpose(2, 0, 1)
            video[frame % self.frames] = img
        video -= 127.5
        video /= 127.5
        return torch.as_tensor(video)
