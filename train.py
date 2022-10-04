from model import Generator, Discriminator
from loss import DiscriminatorLoss, GeneratorLoss
from data_loader import make_loader
import torch as t
from torch import nn, optim
import os
from numpy import random as rd
from utils import imgs_from_tensor
import shutil
import numpy as np
import cv2
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))



def train_epoch(g_criterion, d_criterion, g_optimizer, d_optimizer, train_loader, generator, discriminator, current_epoch):
    global total_step
    steps = len(train_loader)
    current_step = 1
    for real_imgs in train_loader:
        real_imgs_cuda = real_imgs.cuda(device_ids[0])
        generator.eval()
        for i_d in range(d_train_times_per_step):
            noise = t.from_numpy(rd.normal(0, 1, (batch_size // 2, noise_dim))).type(t.FloatTensor).cuda(device_ids[0])
            with t.no_grad():
                fake_imgs_cuda = generator(noise)
            d_loss = d_criterion(fake_imgs_cuda, real_imgs_cuda, discriminator, True)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
        generator.train()
        for i_g in range(g_train_times_per_step):
            noise = t.from_numpy(rd.normal(0, 1, (batch_size, noise_dim))).type(t.FloatTensor).cuda(device_ids[0])
            fake_imgs_cuda = generator(noise)
            g_loss = g_criterion(fake_imgs_cuda, discriminator, real_imgs_cuda)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            d_optimizer.zero_grad()
        if current_step % print_step == 0:
            print("epoch:%d/%d, step:%d/%d, g_loss:%.5f, d_loss:%.5f" % (current_epoch, epoch, current_step, steps, g_loss.item(), d_loss.item()))
        if total_step % save_img_total_step == 0:
            generator.eval()
            with t.no_grad():
                noise = t.from_numpy(rd.normal(0, 1, (row_img_count ** 2, noise_dim))).type(t.FloatTensor).cuda(device_ids[0])
                g_output = generator(noise)
                cv2_imgs = imgs_from_tensor(g_output)
                rows = []
                for i in range(row_img_count):
                    row = np.concatenate(cv2_imgs[i * row_img_count:(i + 1) * row_img_count], axis=1)
                    rows.append(row)
                save_img = np.concatenate(rows, axis=0)
                cv2.imwrite(os.path.join(img_save_dir, "%d.png" % (total_step,)), save_img)
        current_step += 1
        total_step += 1
    print("saving epoch model......")
    t.save(generator.module.state_dict(), "epoch.pth")
    return generator, discriminator


def valid_epoch(g_criterion, d_criterion, generator, discriminator, valid_loader, current_epoch):
    global best_g_loss
    steps = len(valid_loader)
    accum_g_loss = 0
    accum_d_loss = 0
    for real_imgs in valid_loader:
        real_imgs_cuda = real_imgs.cuda(device_ids[0])
        generator.eval()
        with t.no_grad():
            noise = t.from_numpy(rd.normal(0, 1, (batch_size // 2, noise_dim))).type(t.FloatTensor).cuda(device_ids[0])
            fake_imgs_cuda = generator(noise)
            d_loss = d_criterion(fake_imgs_cuda, real_imgs_cuda, discriminator, False)
            noise = t.from_numpy(rd.normal(0, 1, (batch_size, noise_dim))).type(t.FloatTensor).cuda(device_ids[0])
            fake_imgs_cuda = generator(noise)
            g_loss = g_criterion(fake_imgs_cuda, discriminator, real_imgs_cuda)
            accum_d_loss += d_loss.item()
            accum_g_loss += g_loss.item()
    avg_g_loss = accum_g_loss / steps
    avg_d_loss = accum_d_loss / steps
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        print("saving best model......")
        t.save(generator.module.state_dict(), "best.pth")
    print("###########valid epoch:%d#################" % (current_epoch,))
    print("g_loss:%.5f, d_loss:%.5f" % (avg_g_loss, avg_d_loss))
    return generator, discriminator


def main():
    generator = Generator(img_channels, noise_dim, img_size)
    generator = nn.DataParallel(module=generator, device_ids=device_ids)
    generator = generator.cuda(device_ids[0])
    discriminator = Discriminator(img_channels, img_size)
    discriminator = nn.DataParallel(module=discriminator, device_ids=device_ids)
    discriminator = discriminator.cuda(device_ids[0])
    g_optimizer = optim.Adam(params=generator.parameters(), lr=g_init_lr)
    d_optimizer = optim.Adam(params=discriminator.parameters(), lr=d_init_lr)
    g_lr_sch = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epoch, eta_min=g_final_lr)
    d_lr_sch = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epoch, eta_min=d_final_lr)
    for e in range(epoch):
        current_epoch = e + 1
        train_loader = make_loader(data_root_dir, True, img_size, batch_size // 2, num_workers)
        valid_loader = make_loader(data_root_dir, False, img_size, batch_size // 2, num_workers)
        generator, discriminator = train_epoch(g_criterion, d_criterion, g_optimizer, d_optimizer, train_loader, generator, discriminator, current_epoch)
        generator, discriminator = valid_epoch(g_criterion, d_criterion, generator, discriminator, valid_loader, current_epoch)
        d_lr_sch.step()
        g_lr_sch.step()


if __name__ == "__main__":
    epoch = 50
    batch_size = 256
    g_init_lr = 0.01
    g_final_lr = 0.001
    d_init_lr = 0.01
    d_final_lr = 0.001
    feature_loss_weight = 0.75
    bce_loss_weight = 0.25
    g_criterion = GeneratorLoss(feature_loss_weight=feature_loss_weight, bce_loss_weight=bce_loss_weight).cuda(device_ids[0])
    d_criterion = DiscriminatorLoss().cuda(device_ids[0])
    noise_dim = 128
    g_train_times_per_step = 1
    d_train_times_per_step = 1
    data_root_dir = r"F:\data\chapter7\data"
    num_workers = 4
    print_step = 10
    img_channels = 3
    img_size = 96
    best_g_loss = float("inf")
    total_step = 1
    save_img_total_step = 500
    img_count = 25
    img_save_dir = r"images"
    row_img_count = np.sqrt(img_count).astype(int)
    if os.path.exists(img_save_dir):
        shutil.rmtree(img_save_dir)
    os.mkdir(img_save_dir)
    main()