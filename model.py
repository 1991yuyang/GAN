import torch as t
from torch import nn
from torchvision import transforms as T
import numpy as np
from numpy import random as rd
import math


act_dict = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU
}


class Generator(nn.Module):

    def __init__(self, noise_dim, img_size):
        super(Generator, self).__init__()
        layer_count = math.ceil(np.log2(img_size / noise_dim))
        self.change_channels = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2)
        )
        layers = [self.one_layer(2 ** (i + 4), 2 ** (i + 5), "leakyrelu") for i in range(layer_count)]
        self.model = nn.Sequential(
            *layers
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=img_size)
        self.decrease_channels = nn.Sequential(
            nn.Conv2d(in_channels=2 ** (layer_count + 4), out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.change_channels(x)
        x = self.model(x)
        x = self.pool(x)
        x = self.decrease_channels(x)
        return x

    def one_layer(self, in_channels, out_channels, act):
        base = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
        )
        if "leaky" not in act.lower():
            base.add_module("act", act_dict[act]())
            return base
        base.add_module("act", act_dict[act](0.2))
        return base


class Discriminator(nn.Module):

    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.layer1 = self.one_layer(3 * img_size ** 2, 32, "relu", False)
        self.layer2 = self.one_layer(32, 16, "relu", False)
        self.layer3 = self.one_layer(16, 8, "relu", True)
        self.clsf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=8, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x.view((x.size()[0], -1))
        layer1_result = self.layer1(x1)
        layer2_result = self.layer2(layer1_result)
        layer3_result = self.layer3(layer2_result)
        x3 = self.clsf(layer3_result)
        return x3, [layer1_result, layer2_result, layer3_result]

    def one_layer(self, in_features, out_features, act, is_dropout):
        base = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features, bias=False),
                nn.BatchNorm1d(num_features=out_features)
        )
        if is_dropout:
            base.add_module("dropout", nn.Dropout(0.5))
        if "leaky" not in act.lower():
            base.add_module("act", act_dict[act]())
            return base
        base.add_module("act", act_dict[act](0.2))
        return base


if __name__ == "__main__":
    import torch as t
    g = Generator(noise_dim=16, img_size=129)
    d = Discriminator()
    g_input = t.from_numpy(rd.normal(0, 1, (8, 1, 16, 16))).type(t.FloatTensor)
    g_out = g(g_input)
    d_output = d(g_out)
    print(t.mean(-t.log(d_output)))
