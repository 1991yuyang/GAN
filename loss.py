import torch as t
from torch import nn


class GeneratorLoss(nn.Module):

    def __init__(self, feature_loss_weight, bce_loss_weight):
        super(GeneratorLoss, self).__init__()
        self.bce = nn.BCELoss().cuda(0)
        self.feature_loss_weight = feature_loss_weight
        self.bce_loss_weight = bce_loss_weight

    def forward(self, fake_imgs, discriminator, real_imgs):
        discriminator.eval()
        d_output, fake_features = discriminator(fake_imgs)  # (N, 1)
        with t.no_grad():
            _, real_features = discriminator(real_imgs)  # (N, 1)
        target = t.ones(d_output.size()).to(d_output.device)
        loss_bce = self.bce(d_output, target)
        loss_feature = 0
        for i in range(len(real_features)):
            fake_feature = fake_features[i]
            real_feature = real_features[i]
            loss_feature += t.pow(t.norm(t.mean(fake_feature, dim=0) - t.mean(real_feature, dim=0)), 2)
        loss_feature = loss_feature / len(real_features)
        loss = self.feature_loss_weight * loss_feature + self.bce_loss_weight * loss_bce
        return loss


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce = nn.BCELoss().cuda(0)

    def forward(self, fake_imgs, real_imgs, discriminator, is_train):
        if is_train:
            discriminator.train()
        else:
            discriminator.eval()
        d_fake_output, _ = discriminator(fake_imgs)  # (N / 2, 1)
        fake_target = t.zeros(d_fake_output.size()).to(fake_imgs.device)
        d_real_output, _ = discriminator(real_imgs)  # (N / 2, 1)
        real_target = t.ones(d_real_output.size()).to(real_imgs.device)
        d_fake_loss = self.bce(d_fake_output, fake_target)
        d_real_loss = self.bce(d_real_output, real_target)
        loss = (d_fake_loss + d_real_loss) / 2
        return loss