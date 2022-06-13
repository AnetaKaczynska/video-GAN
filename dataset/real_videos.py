import itertools
import os
import re
from random import randrange

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RealVideos(Dataset):
    def __init__(self, root='/shared/results/Skopia/videos24frames', num_frame=24, duration=24):
        self.img_dir = root
        self.num_frame = num_frame
        self.ch, self.duration, self.h, self.w = 3, duration, 256, 256
        if num_frame == 8:
            self.video_names = self._load_videos8()
        else:
            self.video_names = self._load_videos()

        self.indexes = np.linspace(0., num_frame, num=duration, endpoint=False).astype(int)

    def _load_videos(self):
        images = os.listdir(self.img_dir)

        groups = []
        uniquekeys = []
        keyfunc = lambda p: p.split('_')[:2]
        data = sorted(images, key=keyfunc)
        for k, g in itertools.groupby(data, keyfunc):
            groups.append(list(g))  # Store group iterator as a list
            uniquekeys.append(k)

        fgroups = list(filter(lambda g: len(g) == self.num_frame, groups))
        video_names = list(map(lambda x: "_".join(x[0].split('_')[:2]), fgroups))

        return video_names

    def _load_videos8(self):
        images = os.listdir(self.img_dir)
        img_name = re.compile('([0-9]+_[0-9]+)_[0-7]+.jpg')
        video_names = {img_name.search(img).groups()[0] for img in images}
        return list(video_names)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, i):
        if self.duration == self.num_frame:
            indexes = self.indexes
        else:
            # indexes = np.sort(np.random.choice(self.num_frame, self.duration, replace=False))
            indexes = self.indexes + randrange(0, self.num_frame - self.indexes[-1])

        video = np.empty((self.duration, self.ch, self.h, self.w), dtype=np.float32)
        video_name = self.video_names[i]
        for idx, t in enumerate(indexes):
            if self.duration == 8:
                img_path = self.img_dir + f'/{video_name}_{t}.jpg'
                img = Image.open(img_path)
                img = img.resize((self.w, self.h))
            else:
                img_path = self.img_dir + f'/{video_name}_{t + 1}.jpg'
                img = Image.open(img_path)
            # img = Image.open(img_path)
            img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
            video[idx] = img
        video -= 127.5
        video /= 127.5
        return torch.as_tensor(video), indexes


if __name__ == '__main__':
    real_videos = RealVideos()
    print(len(real_videos))
