import torch as t
from torch import nn
from torch.nn import functional as F


class GeneratorLoss(nn.Module):

    def __init__(self):
        super(GeneratorLoss, self).__init__()
        pass

    def forward(self, fake_imgs, discriminator, is_train):
        if is_train:
            discriminator.eval()
        fake_features = discriminator(fake_imgs)  # (N, k)
        loss = 0
        for fake_feature in fake_features:
            loss -= t.mean(t.mean(fake_feature, dim=0))
        loss = loss / len(fake_features)
        return loss


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        pass

    def forward(self, fake_imgs, real_imgs, discriminator, is_train):
        if is_train:
            discriminator.train()
        fake_count = fake_imgs.size()[0]
        images = t.cat([fake_imgs, real_imgs], dim=0)
        features = discriminator(images)
        loss = 0
        for i in range(len(features)):
            feature = features[i]
            fake_feature = feature[:fake_count, ...]
            real_feature = feature[fake_count:, ...]
            loss += (t.mean(t.mean(fake_feature, dim=0) - t.mean(real_feature, dim=0)))
        loss = loss / len(features)
        return loss