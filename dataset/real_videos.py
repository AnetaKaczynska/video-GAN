import os
import re
import itertools
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RealVideos(Dataset):
    def __init__(self):
        self.img_dir = '/shared/results/Skopia/videos24frames'
        self.ch, self.duration, self.h, self.w = 3, 24, 256, 256
        self.video_names = self._load_videos()

    def _load_videos(self):
        images = os.listdir(self.img_dir)
        
        groups = []
        uniquekeys = []
        keyfunc = lambda p: p.split('_')[:2]
        data = sorted(images, key=keyfunc)
        for k, g in itertools.groupby(data, keyfunc):
            groups.append(list(g))      # Store group iterator as a list
            uniquekeys.append(k)

        fgroups = list(filter(lambda g: len(g) == 24, groups))
        video_names = list(map(lambda x: "_".join(x[0].split('_')[:2]), fgroups))

        return video_names

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, i):
        video = np.empty((self.duration, self.ch, self.h, self.w), dtype=np.float32)
        video_name = self.video_names[i]
        for t in range(self.duration):
            img_path = self.img_dir + f'/{video_name}_{t+1}.jpg'
            img = np.asarray(Image.open(img_path), dtype=np.float32).transpose(2, 0, 1)
            video[t] = img
        # mean = np.mean(video, axis=(2, 3), keepdims=True)
        # std = np.std(video, axis=(2, 3), keepdims=True)
        # video = (video - mean) / std
        video -= 127.5
        video /= 127.5
        return torch.as_tensor(video)


if __name__ == '__main__':
    real_videos = RealVideos()
    print(len(real_videos))
