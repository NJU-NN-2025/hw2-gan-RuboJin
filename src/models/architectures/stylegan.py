import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------
# 1) Mapping Network (z → w)
# ---------------------------------------------
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=128, w_dim=128, num_layers=4):
        super().__init__()
        layers = []

        for _ in range(num_layers):
            layers.append(nn.Linear(z_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            z_dim = w_dim

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # Normalize latent input (StyleGAN2 trick)
        z = z / torch.norm(z, dim=1, keepdim=True)
        return self.net(z)



# ---------------------------------------------
# 2) Modulated Conv2D (StyleGAN2 version)
# ---------------------------------------------
class ModulatedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel, style_dim, demodulate=True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel = kernel
        self.demodulate = demodulate

        self.weight = nn.Parameter(
            torch.randn(1, out_c, in_c, kernel, kernel)
        )
        self.style = nn.Linear(style_dim, in_c)
        self.pad = kernel // 2

    def forward(self, x, w):
        batch = x.size(0)

        # style modulation: map w to scale per channel
        s = self.style(w).view(batch, 1, self.in_c, 1, 1)  # (B,1,in,1,1)
        weight = self.weight * (s + 1)  # broadcast

        # demodulation: remove magnitude bias
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum([2,3,4]) + 1e-8)
            weight = weight * d.view(batch, self.out_c, 1, 1, 1)

        # reshape for group conv
        x = x.view(1, batch * self.in_c, x.size(2), x.size(3))
        weight = weight.view(batch * self.out_c, self.in_c, self.kernel, self.kernel)
        out = F.conv2d(x, weight, padding=self.pad, groups=batch)
        return out.view(batch, self.out_c, out.size(2), out.size(3))



# ---------------------------------------------
# 3) Styled Conv Block
#    (ModulatedConv2D + Noise + Activation)
# ---------------------------------------------
class StyledConv(nn.Module):
    def __init__(self, in_c, out_c, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_c, out_c, 3, style_dim)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_c))

    def forward(self, x, w, noise=None):
        out = self.conv(x, w)

        if noise is None:
            noise = torch.randn(out.size(0), 1, out.size(2), out.size(3), device=out.device)
        out = out + self.noise_strength * noise

        out = out + self.bias.view(1, -1, 1, 1)
        return F.leaky_relu(out, 0.2)



# ---------------------------------------------
# 4) ToRGB Layer
# ---------------------------------------------
class ToRGB(nn.Module):
    def __init__(self, in_c, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_c, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x, w, skip=None):
        out = self.conv(x, w)
        out = out + self.bias.view(1, 3, 1, 1)
        if skip is not None:
            out = F.interpolate(skip, scale_factor=2) + out
        return out



# ---------------------------------------------
# 5) StyleGAN2 Generator (Lite)
# ---------------------------------------------
class StyleGAN2Generator(nn.Module):
    def __init__(self, z_dim=128, img_size=64, fmap=64, img_channels=3):
        super().__init__()

        self.z_dim = z_dim
        self.w_dim = z_dim
        self.mapping = MappingNetwork(z_dim, self.w_dim)

        # how many upsampling stages?
        depth = int(math.log2(img_size)) - 2  # because start from 4×4

        self.channels = [min(fmap * (2 ** i), 512) for i in range(depth+1)]
        self.channels.reverse()  # big → small resolution

        # constant input 4×4
        self.const = nn.Parameter(torch.randn(1, self.channels[0], 4, 4))

        # blocks
        self.conv_blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        # first block: 4×4
        self.conv_blocks.append(StyledConv(self.channels[0], self.channels[0], self.w_dim))
        self.to_rgbs.append(ToRGB(self.channels[0], self.w_dim))

        # subsequent blocks: upsample × conv
        in_c = self.channels[0]
        for c in self.channels[1:]:
            self.conv_blocks.append(StyledConv(in_c, c, self.w_dim))
            self.to_rgbs.append(ToRGB(c, self.w_dim))
            in_c = c

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.repeat(z.size(0), 1, 1, 1)

        rgb = None
        for block, to_rgb in zip(self.conv_blocks, self.to_rgbs):
            x = block(x, w)
            rgb = to_rgb(x, w, skip=rgb)

            # upsample feature map for next block
            if block != self.conv_blocks[-1]:
                x = F.interpolate(x, scale_factor=2)

        return torch.tanh(rgb)



# ---------------------------------------------
# 6) Discriminator (lightweight StyleGAN2 D)
# ---------------------------------------------
class StyleGAN2Discriminator(nn.Module):
    def __init__(self, img_size=64, fmap=64, img_channels=3):
        super().__init__()

        depth = int(math.log2(img_size)) - 2
        channels = [min(fmap * (2 ** i), 512) for i in range(depth+1)]

        layers = []
        in_c = img_channels

        # downsample blocks
        for c in channels:
            layers.append(nn.Conv2d(in_c, c, 3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            in_c = c

        self.body = nn.Sequential(*layers)

        final_res = img_size // (2 ** (depth+1))
        self.fc = nn.Linear(in_c * final_res * final_res, 1)

    def forward(self, x):
        x = self.body(x)
        x = x.flatten(1)
        return self.fc(x).view(-1)
