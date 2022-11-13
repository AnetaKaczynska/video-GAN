import itertools
import os
import re
from random import randrange
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class RealVideos(Dataset):
    def __init__(self, img_dir, noise_dir, num_frame=24, duration=24):
        self.img_dir = img_dir
        self.noise_dir = noise_dir
        self.num_frame = num_frame
        self.ch, self.duration, self.h, self.w = 3, duration, 256, 256
        self._load_data()

        self.indexes = np.linspace(0., num_frame, num=duration, endpoint=False).astype(int)

    def _load_data(self):
        images = os.listdir(self.img_dir)
        noises = os.listdir(self.noise_dir)

        keyfunc = lambda p: "_".join(p.split('_')[:2])
        data = defaultdict(dict)
        for k, g in itertools.groupby(sorted(images, key=natural_keys), keyfunc):
            data[k]["images"] = list(g)

        for noise in noises:
            data[noise.split('.')[0]]["noise"] = noise

        data = dict(filter(lambda d: "noise" in d[1] and "images" in d[1], data.items()))
        data = dict(filter(lambda g: len(g[1]["images"]) == self.num_frame, data.items()))
        self.video_names = list(map(lambda x: "_".join(x["images"][0].split('_')[:2]), data.values()))
        self.noise_names = list(map(lambda x: x["noise"], data.values()))
        assert len(self.video_names) == len(self.noise_names)

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
        noise_name = self.noise_names[i]
        noise = torch.load(Path(self.noise_dir) / noise_name)
        for idx, t in enumerate(indexes):
            img_path = self.img_dir + f'/{video_name}_{t}.jpg'
            img = Image.open(img_path)
            img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
            video[idx] = img
        video -= 127.5
        video /= 127.5
        return torch.as_tensor(video), noise, indexes


if __name__ == '__main__':
    real_videos = RealVideos()
    print(len(real_videos))
