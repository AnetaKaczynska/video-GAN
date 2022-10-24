from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image


class RealImages(Dataset):
    def __init__(self, root: Path = Path('/shared/results/Skopia/videos24frames')):
        self.img_dir = root
        self.paths = list(root.iterdir())
        self.ch, self.h, self.w = 3, 256, 256
       
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img_path = self.paths[i]
        img = Image.open(img_path)
        img = img.resize((self.w, self.h))
        img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
        img -= 127.5
        img /= 127.5
        img = torch.as_tensor(img)
        return img, str(img_path)


if __name__ == '__main__':
    real_images = RealImages()
    print(len(real_images))
    print(real_images[10].shape)
