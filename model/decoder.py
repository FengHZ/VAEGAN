from torch import nn


class Decoder(nn.Module):
    def __init__(self, latent_dim=100, num_feature=64, num_channel=3, data_parallel=True):
        super(Decoder, self).__init__()
        decoder = nn.Sequential(
            # input is Z*1*1, going into a convolution
            nn.ConvTranspose2d(latent_dim, num_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature * 8),
            nn.ReLU(True),
            # state size. (num_feature*8) x 4 x 4
            nn.ConvTranspose2d(num_feature * 8, num_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature * 4),
            nn.ReLU(True),
            # state size. (num_feature*4) x 8 x 8
            nn.ConvTranspose2d(num_feature * 4, num_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature * 2),
            nn.ReLU(True),
            # state size. (num_feature*2) x 16 x 16
            nn.ConvTranspose2d(num_feature * 2, num_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature),
            nn.ReLU(True),
            # state size. (num_feature) x 32 x 32
            nn.ConvTranspose2d(num_feature, num_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channel) x 64 x 64
        )
        if data_parallel:
            self.decoder = nn.DataParallel(decoder)
        else:
            self.decoder = decoder
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, input):
        return self.decoder(input)
