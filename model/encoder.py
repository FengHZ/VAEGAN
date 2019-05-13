from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, num_channel=3, num_feature=64, latent_dim=100, data_parallel=True):
        super(Encoder, self).__init__()
        features = nn.Sequential(
            # input is (num_channel) x 64 x 64
            nn.Conv2d(num_channel, num_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_feature) x 32 x 32
            nn.Conv2d(num_feature, num_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_feature*2) x 16 x 16
            nn.Conv2d(num_feature * 2, num_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_feature*4) x 8 x 8
            nn.Conv2d(num_feature * 4, num_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        z_mean_map = nn.Sequential(
            # state size. (num_feature*8) x 4 x 4
            nn.Conv2d(num_feature * 8, latent_dim, 4, 1, 0, bias=True),
        )
        z_log_sigma_map = nn.Sequential(
            # state size. (num_feature*8) x 4 x 4
            nn.Conv2d(num_feature * 8, latent_dim, 4, 1, 0, bias=True),
        )
        if data_parallel:
            self.features = nn.DataParallel(features)
            self.z_mean_map = nn.DataParallel(z_mean_map)
            self.z_log_sigma_map = nn.DataParallel(z_log_sigma_map)
        else:
            self.features = features
            self.z_mean_map = z_mean_map
            self.z_log_sigma_map = z_log_sigma_map
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        x = self.features(x)
        mu = self.z_mean_map(x)
        log_sigma = self.z_mean_map(x)
        sigma = torch.exp(log_sigma)
        std_z = torch.randn(mu.size())
        if mu.is_cuda:
            std_z = std_z.cuda()
        z_sample = mu + std_z * sigma
        return mu, log_sigma, sigma, z_sample
