import torch as t
from torch import nn
from torch.nn import functional as F


class GeneratorLoss(nn.Module):

    def __init__(self, feature_loss_weight, bce_loss_weight, dist_loss_weight, ls_loss_weight):
        super(GeneratorLoss, self).__init__()
        self.bce = nn.BCELoss().cuda(0)
        self.feature_loss_weight = feature_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.dist_loss_weight = dist_loss_weight
        self.ls_loss_weight = ls_loss_weight

    def forward(self, fake_imgs, discriminator, real_imgs, is_train):
        if is_train:
            discriminator.eval()
        d_output, fake_features = discriminator(fake_imgs)  # (N, 1)
        with t.no_grad():
            _, real_features = discriminator(real_imgs)  # (N, 1)
        target = t.ones(d_output.size()).type(t.FloatTensor).to(d_output.device)
        loss_bce = self.bce(d_output, target)
        loss_ls = F.mse_loss(d_output, target)
        loss_feature = 0
        loss_dist = 0
        for i in range(len(real_features)):
            fake_feature = fake_features[i]
            real_feature = real_features[i]
            loss_feature += t.pow(t.norm(t.mean(fake_feature, dim=0) - t.mean(real_feature, dim=0)), 2)
            loss_dist += t.mean(t.exp(-F.pdist(fake_feature, p=2.0)))
        loss_feature = loss_feature / len(real_features)
        loss_dist = loss_dist / len(real_features)
        loss = self.feature_loss_weight * loss_feature + self.bce_loss_weight * loss_bce + self.dist_loss_weight * loss_dist + self.ls_loss_weight * loss_ls
        return loss


class DiscriminatorLoss(nn.Module):

    def __init__(self, label_smooth):
        super(DiscriminatorLoss, self).__init__()
        self.bce = nn.BCELoss().cuda(0)
        self.label_smooth = label_smooth

    def forward(self, fake_imgs, real_imgs, discriminator, is_train):
        if is_train:
            discriminator.train()
        d_fake_output, fake_features = discriminator(fake_imgs)  # (N / 2, 1)
        fake_target = t.zeros(d_fake_output.size()).type(t.FloatTensor).to(fake_imgs.device)
        d_real_output, real_features = discriminator(real_imgs)  # (N / 2, 1)
        real_target = t.ones(d_real_output.size()).type(t.FloatTensor).to(real_imgs.device) - self.label_smooth
        d_fake_bce_loss = self.bce(d_fake_output, fake_target)
        d_real_bce_loss = self.bce(d_real_output, real_target)
        d_fake_ls_loss = F.mse_loss(d_fake_output, fake_target)
        d_real_ls_loss = F.mse_loss(d_real_output, real_target)
        loss_ls = (d_fake_ls_loss + d_real_ls_loss) / 2
        loss_bce = (d_fake_bce_loss + d_real_bce_loss) / 2
        loss = (loss_ls + loss_bce) / 2
        return loss