import torch.nn as nn
import math

class LSGANGenerator(nn.Module):
    """Least Squares GAN Generator"""
    def __init__(self, z_dim=128, img_channels=3, feature_maps=64, img_size=64):
        super().__init__()
        n_layers = int(math.log2(img_size) - 3)
        layers = []

        in_channels = z_dim
        out_channels = feature_maps * (2 ** n_layers)

        # 第一层: latent -> 小特征图
        layers += [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]

        # 中间层: 上采样
        for i in range(n_layers, 0, -1):
            layers += [
                nn.ConvTranspose2d(
                    feature_maps * (2 ** i),
                    feature_maps * (2 ** (i - 1)),
                    4, 2, 1, bias=False
                ),
                nn.BatchNorm2d(feature_maps * (2 ** (i - 1))),
                nn.ReLU(True)
            ]

        # 输出层
        layers += [
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class LSGANDiscriminator(nn.Module):
    """Least Squares GAN Discriminator"""
    def __init__(self, img_channels=3, feature_maps=64, img_size=64):
        super().__init__()
        n_layers = int(math.log2(img_size) - 3)
        layers = []

        # 第一层
        layers += [
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # 中间层
        for i in range(1, n_layers + 1):
            layers += [
                nn.Conv2d(
                    feature_maps * (2 ** (i - 1)),
                    feature_maps * (2 ** i),
                    4, 2, 1, bias=False
                ),
                nn.BatchNorm2d(feature_maps * (2 ** i)),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # 输出层（不加 sigmoid）
        final_kernel = img_size // (2 ** (n_layers + 1))
        layers += [
            nn.Conv2d(feature_maps * (2 ** n_layers), 1, final_kernel, 1, 0, bias=False)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.view(-1)
