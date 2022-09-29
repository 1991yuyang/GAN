from torch.utils import data
from torchvision import transforms as T
import os
from PIL import Image
import torch as t
import cv2
import numpy as np
"""
data_root_dir
    train
        img1.jpg
        img2.jpg
        ......
    val
        img1.jpg
        img2.jpg
        ......
"""


class MySet(data.Dataset):

    def __init__(self, data_root_dir, is_train, img_size):
        if is_train:
            self.img_dir = os.path.join(data_root_dir, "train")
        else:
            self.img_dir = os.path.join(data_root_dir, "val")
        self.img_names = os.listdir(self.img_dir)
        self.totensor = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        self.img_size = img_size

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_pth = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, axes=[2, 0, 1])
        img_tensor = t.tensor((img - 127.5) / 127.5).type(t.FloatTensor)
        return img_tensor

    def __len__(self):
        return len(self.img_names)


def make_loader(data_root_dir, is_train, batch_size, num_workers, img_size):
    loader = iter(data.DataLoader(MySet(data_root_dir, is_train, img_size), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader