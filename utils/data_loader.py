import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])


class InferRGB(data.Dataset):
    def __init__(self, image_root="", size=320, return_size=False):
        self.return_size = return_size
        self.size = size
        self.images_path = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                            f.endswith('.jpg') or f.endswith('.png')]
        self.images_path = sorted(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        name = os.path.basename(image_path)

        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.float32)
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        # image = np.expand_dims(image, 0)

        size = (w, h)
        out_dict = {}
        out_dict["image"] = image
        out_dict["name"] = name
        if self.return_size:
            out_dict["size"] = size
        return out_dict

    def __len__(self):
        return len(self.images_path)
