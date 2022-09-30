from model import Generator
from torchvision import transforms as T
import torch as t
import os
from numpy import random as rd
import numpy as np
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_size = 96
noise_dim = 16
use_best_model = True
batch_size = 8


def to_bgr(one_output):
    img = np.transpose((one_output.numpy() * 127.5 + 127.5).astype(np.uint8), axes=[1, 2, 0])
    return img


def load_generator():
    model = Generator(noise_dim, img_size)
    if use_best_model:
        model.load_state_dict(t.load("generator_best.pth"))
    else:
        model.load_state_dict(t.load("generator_epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


def inference(model):
    generator_input = t.from_numpy(rd.normal(0, 1, (batch_size, noise_dim ** 2))).type(t.FloatTensor).cuda(0)
    generator_input = generator_input.view((batch_size, 1, noise_dim, noise_dim))
    with t.no_grad():
        output = model(generator_input).cpu().detach()
        pils = [to_bgr(o) for o in output]
    for i, pil in enumerate(pils):
        cv2.imshow("%d" % (i,), pil)
        cv2.waitKey()


if __name__ == "__main__":
    model = load_generator()
    inference(model)