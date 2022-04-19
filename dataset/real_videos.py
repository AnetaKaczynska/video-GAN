import os
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RealVideos(Dataset):
    def __init__(self):
        self.img_dir = '/shared/results/Skopia/videos_8frames'
        self.ch, self.duration, self.h, self.w = 3, 8, 1024, 1024
        self.video_names = self._load_videos()

    def _load_videos(self):
        images = os.listdir(self.img_dir)
        img_name = re.compile('([0-9]+_[0-9]+)_[0-7]+.jpg')
        video_names = {img_name.search(img).groups()[0] for img in images}
        return list(video_names)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, i):
        video = np.empty((self.duration, self.ch, self.h, self.w), dtype=np.float32)
        video_name = self.video_names[i]
        for t in range(self.duration):
            img_path = self.img_dir + f'/{video_name}_{t}.jpg'
            img = np.asarray(Image.open(img_path), dtype=np.uint8).transpose(2, 0, 1)
            video[t] = img
        return torch.as_tensor(video)


if __name__ == '__main__':
    real_videos = RealVideos()
    print(len(real_videos))
