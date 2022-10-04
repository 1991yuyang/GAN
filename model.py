import torch as t
from torch import nn


act = {
    "leaky": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "linear": nn.Sequential,
    "sigmoid": nn.Sigmoid
}
generator_features = [64, 128]
discriminator_features = generator_features[::-1]


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


class Generator(nn.Module):

    def __init__(self, img_channels, noise_dim, img_size):
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

    def __init__(self, img_channels, img_size):
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
        self.fcs.add_module("last_fc", FC(in_features=discriminator_features[-1], out_features=1, is_bn=False, act_name="sigmoid"))

    def forward(self, x):
        features = []
        x = x.view((x.size()[0], -1))
        for n, m in self.fcs._modules.items():
            x = m(x)
            features.append(x)
        return x, features


if __name__ == "__main__":
    g_input = t.randn(2, 128)
    g = Generator(img_channels=3, noise_dim=128, img_size=256)
    g_out = g(g_input)
    d = Discriminator(img_channels=3, img_size=256)
    d_output = d(g_out)
    print(d_output.size())
    from loss import GeneratorLoss, DiscriminatorLoss
    g_criterion = GeneratorLoss()
    gloss = g_criterion(g_out, d)
    d_criterion = DiscriminatorLoss()
    dloss = d_criterion(g_out, g_out, d)
    print(dloss)