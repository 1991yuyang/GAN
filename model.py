import math

import torch as t
from torch import nn
from torch.nn import functional as F

act = {
    "leaky": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "linear": nn.Sequential,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU
}


class FC(nn.Module):

    def __init__(self, in_features, out_features, is_bn, act_name):
        super(FC, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=not is_bn)
        )
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm1d(num_features=out_features))
        if act_name.lower() == "leaky":
            self.block.add_module("act", act[act_name](0.2))
        else:
            self.block.add_module("act", act[act_name]())

    def forward(self, x):
        ret = self.block(x)
        return ret


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, is_bn, act_name):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2 - 1, bias=not is_bn))
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        if act_name.lower() == "leaky":
            self.block.add_module("act", act[act_name](0.2))
        else:
            self.block.add_module("act", act[act_name]())

    def forward(self, x):
        x = self.block(x)
        return x


class DeConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, is_bn, act_name):
        super(DeConvBlock, self).__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2 - 1, bias=not is_bn))
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        if act_name.lower() == "leaky":
            self.block.add_module("act", act[act_name](0.2))
        else:
            self.block.add_module("act", act[act_name]())

    def forward(self, x):
        x = self.block(x)
        return x


class Generator(nn.Module):

    def __init__(self, img_channels, noise_dim, img_size, generator_features):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.fcs = nn.Sequential()
        for i, features in enumerate(generator_features):
            if i == 0:
                in_features = noise_dim
            else:
                in_features = generator_features[i - 1]
            out_features = features
            self.fcs.add_module("fc_%d" % (i,), FC(in_features=in_features, out_features=out_features, is_bn=True, act_name="leaky"))
        self.fcs.add_module("last_fc", FC(in_features=generator_features[-1], out_features=img_channels * img_size * img_size, is_bn=False, act_name="tanh"))

    def forward(self, x):
        fc_result = self.fcs(x)
        ret = fc_result.view((fc_result.size()[0], self.img_channels, self.img_size, self.img_size))
        return ret


class Discriminator(nn.Module):

    def __init__(self, img_channels, img_size, discriminator_features):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.fcs = nn.Sequential()
        for i, features in enumerate(discriminator_features):
            if i == 0:
                in_features = img_channels * img_size ** 2
            else:
                in_features = discriminator_features[i - 1]
            out_features = features
            self.fcs.add_module("fc_%d" % (i,), FC(in_features=in_features, out_features=out_features, is_bn=True, act_name="leaky"))
        self.fcs.add_module("last", FC(in_features=discriminator_features[-1], out_features=1, is_bn=True, act_name="leaky"))

    def forward(self, x):
        fc_features = []
        x = x.view((x.size()[0], -1))
        for n, m in self.fcs._modules.items():
            x = m(x)
            fc_features.append(x)  # (N, k)
        return fc_features


class CNNGenerator(nn.Module):

    def __init__(self, img_channels, noise_dim, img_size, generator_features):
        super(CNNGenerator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.us = nn.Sequential()
        for i, f in enumerate(generator_features):
            if i == 0:
                in_channels = noise_dim
            else:
                in_channels = generator_features[i - 1]
            self.us.add_module("deconv_%d" % (i, ), DeConvBlock(in_channels=in_channels, out_channels=f, kernel_size=4, is_bn=True, act_name="leaky"))
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=generator_features[-1], out_channels=img_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.us(x)
        x = self.last(x)
        return x


class CNNDiscriminator(nn.Module):

    def __init__(self, img_channels, img_size, discriminator_features):
        super(CNNDiscriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.conv = nn.Sequential()
        for i, f in enumerate(discriminator_features):
            if i == 0:
                in_channels = img_channels
            else:
                in_channels = discriminator_features[i - 1]
            self.conv.add_module("conv_%d" % (i,), ConvBlock(in_channels=in_channels, out_channels=f, kernel_size=4, is_bn=True, act_name="leaky"))
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=discriminator_features[-1], out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        features = []
        for n, m in self.conv._modules.items():
            x = m(x)
            features.append(x.view((x.size()[0], -1)))
        last_feature = self.last(x)
        features.append(last_feature.view((x.size()[0], -1)))
        return features


if __name__ == "__main__":
    g_input = t.randn(2, 128, 1, 1)
    g = CNNGenerator(img_channels=3, noise_dim=128, img_size=128, generator_features=[64, 128, 256])
    g_out = g(g_input)
    print(g_out.size())
    d = CNNDiscriminator(img_channels=3, img_size=256, discriminator_features=[256, 128, 64])
    features = d(g_out)
    print(features[-1].size())