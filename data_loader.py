from torch.utils import data
import cv2
import os
import numpy as np
import torch as t

"""
data_root_dir
    train
        1.jpg
        2.jpg
        ......
    val
        1.jpg
        2.jpg
        ......
"""


class MySet(data.Dataset):

    def __init__(self, data_root_dir, is_train, img_size):
        if is_train:
            img_dir = os.path.join(data_root_dir, "train")
        else:
            img_dir = os.path.join(data_root_dir, "val")
        img_names = os.listdir(img_dir)
        self.img_pths = [os.path.join(img_dir, name) for name in img_names]
        self.img_size = img_size

    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = (img - 127.5) / 127.5
        img = np.transpose(img, axes=[2, 0, 1])
        img = t.from_numpy(img).type(t.FloatTensor)
        return img

    def __len__(self):
        return len(self.img_pths)


def make_loader(data_root_dir, is_train, img_size, batch_size, num_workers):
    loader = iter(data.DataLoader(MySet(data_root_dir, is_train, img_size), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    data_root_dir = r"F:\data\chapter7\data"
    is_train = True
    img_size = 256
    batch_size = 8
    num_workers = 4
    loader = make_loader(data_root_dir, is_train, img_size, batch_size, num_workers)
    for img in loader:
        print(img.size())
        print(t.max(img))
        print(t.min(img))
        print("=====================")