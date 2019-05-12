from torch import nn


class Discriminator(nn.Module):
    def __init__(self, num_channel=3, num_feature=64, data_parallel=True):
        super(Discriminator, self).__init__()
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
        classifier = nn.Sequential(
            # state size. (num_feature*8) x 4 x 4
            nn.Conv2d(num_feature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        if data_parallel:
            self.features = nn.DataParallel(features)
            self.classifier = nn.DataParallel(classifier)
        else:
            self.features = features
            self.classifier = classifier
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, input):
        features = self.features(input)
        classification = self.classifier(features)
        return features, classification
