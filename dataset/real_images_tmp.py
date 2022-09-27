import os
import re

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RealImages(Dataset):
    def __init__(self):
        self.image_dir = '/shared/results/Skopia/videos_24frames'
        self.h, self.w = 256, 256
        self.image_names = self._load_images()

    def _load_images(self):
        # images = os.listdir(self.image_dir)
        # image_name = re.compile('([0-9]+_[0-9]+).jpg')
        # image_names = {image_name.search(image).groups()[0] for image in images}
        # return list(image_names)
        return [f'19180_11709_{i}' for i in range(1, 17)]   # [f'19513_16705_{i}' for i in range(1, 17)]   # [f'19106_13529_{i}' for i in range(1, 17)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i):
        image_name = self.image_names[i]

        image_path = self.image_dir + f'/{image_name}.jpg'
        image = Image.open(image_path).convert('RGB')
        # image = image.resize((self.h, self.w), Image.LANCZOS)
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)

        image -= 127.5
        image /= 127.5
        return image_name, torch.as_tensor(image)


if __name__ == '__main__':
    real_images = RealImages()
    for img in real_images:
        pass
