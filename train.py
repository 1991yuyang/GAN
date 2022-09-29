from model import Generator, Discriminator
from loss import GeneratorLoss, DiscriminatorLoss
from data_loader import make_loader
import os
import torch as t
from torch import nn, optim
CUDA_VISIBLE_DEVICES = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
discriminator_train_times_every_step = 1  # 每一个step判别器训练的次数
generator_train_times_every_step = 1  # 每一个step生成器训练的次数
discriminator_train_fake_accu_tresh = 1  # 判别器在生成器生成的假数据上准确率低于此阈值则对判别器进行训练，否则不训练
discriminator_init_lr = 0.01
discriminator_final_lr = 0.0001
generator_init_lr = 0.01
generator_final_lr = 0.0001
batch_size = 1024
noise_dim = 16  # 产生的噪声的维度为[noise_dim, noise_dim]
print_step = 10
epoch = 1000
img_size = 96
num_workers = 8
label_smooth_eta = 0.1
generator_feature_loss_weight = 0.5  # 真样本和假样本判别器特征间距离损失
generator_bce_loss_weight = 0.5  # bce损失
d_weight_range = [-0.5, 0.5]  # 对discriminator的网络权重进行裁剪，表示裁剪范围
data_root_dir = r"/home/guest/yuyang/data/cartoon"
generator_best_loss = float("inf")
discriminator_criterion = DiscriminatorLoss(batch_size, noise_dim, label_smooth_eta).cuda(device_ids[0])
generator_criterion = GeneratorLoss(batch_size, noise_dim, label_smooth_eta, generator_feature_loss_weight, generator_bce_loss_weight).cuda(device_ids[0])


def clip_d_weight(discriminator):
    for name, parameter in discriminator.named_parameters():
        t.clamp_(parameter.data, d_weight_range[0], d_weight_range[1])
    return discriminator


def train_epoch(discriminator, generator, train_loader, generator_criterion, discriminator_criterion, generator_optimizer, discriminator_optimizer, current_epoch):
    steps = len(train_loader)
    current_step = 1
    for train_true_img in train_loader:
        train_true_img_cuda = train_true_img.cuda(device_ids[0])
        for i in range(discriminator_train_times_every_step):
            discriminator_optimizer.zero_grad()
            d_loss, d_real_accu, d_fake_accu = discriminator_criterion(generator, discriminator, train_true_img_cuda, True)
            if d_fake_accu <= discriminator_train_fake_accu_tresh:
                d_loss.backward()
                discriminator_optimizer.step()
                discriminator = clip_d_weight(discriminator)
            else:
                discriminator_optimizer.zero_grad()
        for i in range(generator_train_times_every_step):
            generator_optimizer.zero_grad()
            g_loss, g_fake_accu = generator_criterion(generator, discriminator, True, train_true_img_cuda)
            g_loss.backward()
            generator_optimizer.step()
        if current_step % print_step == 0:
            print("epoch:%d/%d, step:%d/%d, g_loss:%.5f, d_loss:%.5f, d_real_accu:%.5f, d_fake_accu:%.5f, g_fake_accu:%.5f" % (current_epoch, epoch, current_step, steps, g_loss.item(), d_loss.item(), d_real_accu, d_fake_accu, g_fake_accu))
        current_step += 1
    print("saving epoch model......")
    t.save(generator.module.state_dict(), "generator_epoch.pth")
    return generator, discriminator


def valid_epoch(discriminator, generator, valid_loader, generator_criterion, discriminator_criterion, current_epoch):
    global generator_best_loss
    steps = len(valid_loader)
    accum_generator_loss = 0
    accum_discriminator_loss = 0
    accum_d_real_accu = 0
    accum_d_fake_accu = 0
    accum_g_fake_accu = 0
    for valid_true_img in valid_loader:
        valid_true_img_cuda = valid_true_img.cuda(device_ids[0])
        g_loss, g_fake_accu = generator_criterion(generator, discriminator, False, valid_true_img_cuda)
        d_loss, d_real_accu, d_fake_accu = discriminator_criterion(generator, discriminator, valid_true_img_cuda, False)
        accum_d_real_accu += d_real_accu
        accum_d_fake_accu += d_fake_accu
        accum_generator_loss += g_loss.item()
        accum_g_fake_accu += g_fake_accu
        accum_discriminator_loss += d_loss.item()
    generator_avg_loss = accum_generator_loss / steps
    discriminator_avg_loss = accum_discriminator_loss / steps
    avg_d_real_accu = accum_d_real_accu / steps
    avg_d_fake_accu = accum_d_fake_accu / steps
    avg_g_fake_accu = accum_g_fake_accu / steps
    if generator_avg_loss < generator_best_loss:
        generator_best_loss = generator_avg_loss
        print("saving best model......")
        t.save(generator.module.state_dict(), "generator_best.pth")
    print("##########valid epoch:%d#############" % (current_epoch,))
    print("g_loss:%.5f, d_loss:%.5f, d_real_accu:%.5f, d_fake_accu:%.5f, g_fake_accu:%.5f" % (generator_avg_loss, discriminator_avg_loss, avg_d_real_accu, avg_d_fake_accu, avg_g_fake_accu))
    return generator, discriminator


def main():
    generator = Generator(noise_dim, img_size)
    generator = nn.DataParallel(module=generator, device_ids=device_ids)
    generator = generator.cuda(device_ids[0])
    discriminator = Discriminator(img_size)
    discriminator = nn.DataParallel(module=discriminator, device_ids=device_ids)
    discriminator = discriminator.cuda(device_ids[0])
    generator_optimizer = optim.RMSprop(params=generator.parameters(), lr=generator_init_lr)
    discriminator_optimizer = optim.SGD(params=discriminator.parameters(), lr=discriminator_init_lr)
    lr_sch_generator = optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=epoch, eta_min=generator_final_lr)
    lr_sch_discriminator = optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=epoch, eta_min=discriminator_final_lr)
    for e in range(epoch):
        current_epoch = e + 1
        train_loader = make_loader(data_root_dir, True, batch_size // 2, num_workers, img_size)
        valid_loader = make_loader(data_root_dir, False, batch_size // 2, num_workers, img_size)
        generator, discriminator = train_epoch(discriminator, generator, train_loader, generator_criterion, discriminator_criterion, generator_optimizer, discriminator_optimizer, current_epoch)
        generator, discriminator = valid_epoch(discriminator, generator, valid_loader, generator_criterion, discriminator_criterion, current_epoch)
        lr_sch_generator.step()
        lr_sch_discriminator.step()


if __name__ == "__main__":
    main()