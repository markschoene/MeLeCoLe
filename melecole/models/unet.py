"""
This implementation follows https://amaarora.github.io/2020/09/13/unet.html
"""
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import rand
import time
from models.utils import get_norm
from models.modules import Upsampling


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, norm, kernel_size, padding='constant'):
        super(ResidualBlock, self).__init__()

        self.norm1 = get_norm(norm, num_channels)
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=kernel_size, stride=(1, 1, 1),
                               padding='same', padding_mode=padding)
        self.norm2 = get_norm(norm, num_channels)
        self.conv2 = nn.Conv3d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=kernel_size, stride=(1, 1, 1),
                               padding='same', padding_mode=padding)

    def forward(self, x):
        z = self.conv1(F.relu(self.norm1(x)))
        z = self.conv2(F.relu(self.norm2(z)))
        return x + z


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm, padding='constant', drop_last=False):
        super().__init__()
        self.norm1 = get_norm(norm, num_channels=in_channels)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                               padding='same', padding_mode=padding)
        self.res_block = ResidualBlock(num_channels=out_channels, norm=norm, kernel_size=(3, 3, 3), padding=padding)
        self.norm2 = get_norm(norm, num_channels=out_channels)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                               padding='same', padding_mode=padding) if not drop_last else nn.Identity()

    def forward(self, x):
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.res_block(x)
        return self.conv2(F.relu(self.norm2(x)))


class Encoder(nn.Module):
    def __init__(self, channels, norm, stem_kernel_size, stem_stride, stem_padding, padding='constant'):
        super().__init__()

        self.stem = nn.Conv3d(1, channels[0],
                              kernel_size=stem_kernel_size,
                              stride=stem_stride,
                              padding=stem_padding, padding_mode=padding)
        self.stem_norm = get_norm(norm, channels[0])

        self.enc_blocks = nn.ModuleList(
            [Block(channels[0], channels[0], norm=norm, padding=padding)] +
            [Block(channels[i], channels[i + 1], norm=norm, padding=padding)
             for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.out_channels = channels[-1]

    def forward(self, x):
        ftrs = []
        x = self.stem_norm(self.stem(x))
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, channels, norm, upsampling, padding='constant'):
        super().__init__()
        self.channels = channels

        upconv_kernel = (1, 2, 2) if upsampling == 'ConvTranspose3d' else (1, 3, 3)
        self.upconvs = nn.ModuleList(
            [Upsampling(class_name=upsampling,
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=upconv_kernel,
                        scale_factor=(1, 2, 2),
                        padding_mode=padding)
             for i in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i + 1], channels[i + 1], norm=norm, padding=padding)
             for i in range(len(channels) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            x = x + encoder_features[i]
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    # TODO: implement upsampling defaults - ResizeConv prevents checkerboard!
    def __init__(self,
                 feature_channels=(32, 64, 128, 256),
                 stem_kernel_size=(1, 5, 5),
                 stem_stride=(1, 1, 1),
                 stem_padding=(0, 2, 2),
                 norm='instance_norm',
                 padding='replicate',
                 zero_init_residual=True):
        super().__init__()

        # Backbone Architecture
        self.feature_channels = feature_channels
        encoder_channels = feature_channels
        decoder_channels = feature_channels[::-1]
        self.out_channels = decoder_channels[-1]

        self.encoder = Encoder(channels=encoder_channels,
                               norm=norm,
                               stem_kernel_size=stem_kernel_size,
                               stem_stride=stem_stride,
                               stem_padding=stem_padding,
                               padding=padding)
        self.decoder = Decoder(channels=decoder_channels,
                               norm=norm,
                               padding=padding,
                               upsampling='ResizeConv3d')

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # Do this as well for the inter-block UNet skip connections
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    if isinstance(m.norm2, nn.BatchNorm3d):
                        nn.init.constant_(m.norm2.weight, 0)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        return out


if __name__ == "__main__":
    print("creating network ...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = rand(1, 1, 20, 128, 128).to(device)
    net = UNet(norm="batch_norm").to(device)
    print(net)
    print("... network created")
    print('number of trainable parameters', sum(p.numel() for p in net.parameters() if p.requires_grad))

    start = time.time()
    net(t)
    print('time for forward pass', time.time() - start)
    print("Success")
