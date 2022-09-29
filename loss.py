import torch as t
from torch import nn
import numpy as np
from numpy import random as rd


class GeneratorLoss(nn.Module):

    def __init__(self, batch_size, noise_dim, label_smooth_eta, feature_loss_weight, bce_loss_weight):
        super(GeneratorLoss, self).__init__()
        self.batch_size_half = batch_size // 2
        self.noise_dim = noise_dim
        self.bce = nn.BCELoss().cuda(0)
        self.label_smooth_eta = label_smooth_eta
        self.feature_loss_weight = feature_loss_weight
        self.bce_loss_weight = bce_loss_weight

    def forward(self, generator, discriminator, is_train, true_img):
        """

        :param generator_output: [N, 3, img_size, img_size]
        :return:
        """
        discriminator.eval()
        if is_train:
            generator.train()
        else:
            generator.eval()
        generator_input = t.from_numpy(rd.normal(0, 1, (self.batch_size_half, self.noise_dim ** 2))).type(t.FloatTensor).cuda(0)
        generator_input = generator_input.view((self.batch_size_half, 1, self.noise_dim, self.noise_dim))
        if is_train:
            generator_output = generator(generator_input)  # [N, 3, img_size, img_size]
            d_output_fake, features_fake = discriminator(generator_output)  # [N, 1]
            with t.no_grad():
                d_output_real, features_real = discriminator(true_img)
            d_output_fake = d_output_fake.view((-1,))
            loss_bce = self.bce(d_output_fake, t.ones([self.batch_size_half]).cuda(0) - self.label_smooth_eta)
            loss_feature = 0
            for i in range(len(features_fake)):
                feature_fake = features_fake[i]
                feature_real = features_real[i]
                loss_feature += t.pow(t.norm(t.mean(feature_fake, dim=0, keepdim=True) - t.mean(feature_real, dim=0, keepdim=True), dim=1), 2)
            loss_feature = loss_feature / len(features_fake)
            loss = self.bce_loss_weight * loss_bce + self.feature_loss_weight * loss_feature
            g_fake_accu = (d_output_fake < 0.5).sum().item() / d_output_fake.size()[0]
        else:
            with t.no_grad():
                generator_output = generator(generator_input)  # [N, 3, img_size, img_size]
                d_output_fake, features_fake = discriminator(generator_output)  # [N, 1]\
                with t.no_grad():
                    d_output_real, features_real = discriminator(true_img)
                d_output_fake = d_output_fake.view((-1,))
                loss_bce = self.bce(d_output_fake, t.ones([self.batch_size_half]).cuda(0) - self.label_smooth_eta)
                loss_feature = 0
                for i in range(len(features_fake)):
                    feature_fake = features_fake[i]
                    feature_real = features_real[i]
                    loss_feature += t.norm(
                        t.mean(feature_fake, dim=0, keepdim=True) - t.mean(feature_real, dim=0, keepdim=True), dim=1)
                loss_feature = loss_feature / len(features_fake)
                loss = self.bce_loss_weight * loss_bce + self.feature_loss_weight * loss_feature
                g_fake_accu = (d_output_fake < 0.5).sum().item() / d_output_fake.size()[0]
        return loss, g_fake_accu


class DiscriminatorLoss(nn.Module):

    def __init__(self, batch_size, noise_dim, label_smooth_eta):
        super(DiscriminatorLoss, self).__init__()
        self.batch_size_half = batch_size // 2
        self.noise_dim = noise_dim
        self.bce = nn.BCELoss().cuda(0)
        self.label_smooth_eta = label_smooth_eta

    def forward(self, generator, discriminator, true_img, is_train):
        """

        :param generator:
        :param true_img: [N, 3, img_size, img_size]
        :param discriminator:
        :return:
        """
        generator.eval()
        if is_train:
            discriminator.train()
        else:
            discriminator.eval()
        generator_input = t.from_numpy(rd.normal(0, 1, (self.batch_size_half, self.noise_dim ** 2))).type(t.FloatTensor).cuda(0)
        generator_input = generator_input.view((self.batch_size_half, 1, self.noise_dim, self.noise_dim))
        with t.no_grad():
            generator_output = generator(generator_input)
        concate_result = t.cat([true_img, generator_output], dim=0)
        target = t.cat([t.ones([self.batch_size_half]) - self.label_smooth_eta, t.zeros([self.batch_size_half]) + self.label_smooth_eta], dim=0).cuda(0)
        if is_train:
            discriminator_output, _ = discriminator(concate_result)
            discriminator_output = discriminator_output.view((-1,))
            loss = self.bce(discriminator_output, target)
            d_real_accu = self.calc_accu(discriminator_output[:self.batch_size_half], target[:self.batch_size_half])
            d_fake_accu = self.calc_accu(discriminator_output[self.batch_size_half:], target[self.batch_size_half:])
            # loss_fake_part = t.mean(-t.log(1 - discriminator(generator_output) + 0.00000000001))
            # loss_true_part = t.mean(-t.log(discriminator(true_img) + 0.00000000001))
            # loss = (loss_fake_part + loss_true_part) / 2
        else:
            with t.no_grad():
                discriminator_output, _ = discriminator(concate_result)
                discriminator_output = discriminator_output.view((-1,))
                loss = self.bce(discriminator_output, target)
                d_real_accu = self.calc_accu(discriminator_output[:self.batch_size_half], target[:self.batch_size_half])
                d_fake_accu = self.calc_accu(discriminator_output[self.batch_size_half:], target[self.batch_size_half:])
                # loss_fake_part = t.mean(-t.log(1 - discriminator(generator_output) + 0.00000000001))
                # loss_true_part = t.mean(-t.log(discriminator(true_img) + 0.00000000001))
                # loss = (loss_fake_part + loss_true_part) / 2
        return loss, d_real_accu, d_fake_accu

    def calc_accu(self, d_output, target):
        accu = ((d_output > 0.5) == (target > 0.5)).sum().item() / d_output.size()[0]
        return accu