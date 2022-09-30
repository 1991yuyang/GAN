from torch.utils import data
from torchvision import transforms as T
import os
import torch as t
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
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


class MNIST(data.Dataset):

    def __init__(self, data_root_dir, is_train, img_size):
        data_mnist = fetch_openml('mnist_784', version=1, cache=True)
        x, y = data_mnist["data"].values, data_mnist["target"].values
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=32, test_size=0.25)
        self.x_train = x_train.reshape((x_train.shape[0], 28, 28))
        self.x_valid = x_valid.reshape((x_valid.shape[0], 28, 28))
        self.img_size = img_size
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            img = self.x_train[index].astype(np.uint8)
        else:
            img = self.x_valid[index].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, axes=[2, 0, 1])
        img_tensor = t.tensor((img - 127.5) / 127.5).type(t.FloatTensor)
        return img_tensor

    def __len__(self):
        if self.is_train:
            return self.x_train.shape[0]
        return self.x_valid.shape[0]


def make_loader(data_root_dir, is_train, batch_size, num_workers, img_size, use_mnist):
    if use_mnist:
        s = MNIST
    else:
        s = MySet
    loader = iter(data.DataLoader(s(data_root_dir, is_train, img_size), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    s = MNIST(True, 256)
    s[1]