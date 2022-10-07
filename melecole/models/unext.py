"""
This implementation follows https://amaarora.github.io/2020/09/13/unet.html
"""
import torch.cuda
import torch.nn as nn
from torch import rand
from collections import namedtuple
import time
from models.utils import get_norm
from models.modules import Upsampling, Downsampling

OutputTuple = namedtuple('OutputTuple', 'embeddings boundaries')


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    Inspired by: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self,
                 in_channels,
                 norm,
                 expansion=4,
                 kernel_size=(3, 7, 7),
                 padding='replicate'):

        super().__init__()

        # depthwise convolution
        self.dwconv = nn.Conv3d(in_channels, in_channels,
                                kernel_size=kernel_size,
                                padding='same', padding_mode=padding,
                                groups=in_channels)
        self.norm = get_norm(norm, in_channels)

        # voxelwise convolution
        self.pwconv1 = nn.Conv3d(in_channels, int(expansion * in_channels),
                                 kernel_size=(1, 1, 1),
                                 padding='same', padding_mode=padding)
        self.act = nn.ReLU()

        self.pwconv2 = nn.Conv3d(int(expansion * in_channels), in_channels,
                                 kernel_size=(1, 1, 1),
                                 padding='same', padding_mode=padding)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        return x + input


class Encoder(nn.Module):
    def __init__(self, channels, num_blocks, expansion, stem_kernel_size, stem_stride, stem_padding, norm, padding='replicate'):
        super().__init__()
        assert len(channels) == len(num_blocks), "number of channels must equal list entries in num_blocks"

        self.stages = nn.ModuleList()
        self.downsampling = nn.ModuleList()

        self.stem = nn.Conv3d(1, channels[0],
                              kernel_size=stem_kernel_size,
                              stride=stem_stride,
                              padding=stem_padding, padding_mode=padding)
        self.stem_norm = get_norm(norm, channels[0])

        # initialize encoder blocks
        for i in range(len(channels) - 1):
            self.stages.append(nn.ModuleList(
                [ConvNeXtBlock(in_channels=channels[i], norm=norm, padding=padding, expansion=expansion)
                 for _ in range(num_blocks[i])]
            ))
            self.downsampling.append(Downsampling(in_channels=channels[i],
                                                  out_channels=channels[i + 1],
                                                  norm=norm,
                                                  pool=False))

        self.stages.append(nn.ModuleList(
            [ConvNeXtBlock(channels[-1], norm=norm, padding=padding,  expansion=expansion)
             for _ in range(num_blocks[-1])]
        ))

    def forward(self, x):
        ftrs = []

        # initial stem computation
        x = self.stem_norm(self.stem(x))

        # compute the stages
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            ftrs.append(x)
            if i < len(self.downsampling):
                x = self.downsampling[i](x)

        # output the list of stage final results
        return ftrs


class Decoder(nn.Module):
    def __init__(self, channels, num_blocks, expansion, norm, upsampling, padding='replicate'):
        super().__init__()

        self.stages = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        # initialize decoder blocks
        for i in range(1, len(channels)):
            self.stages.append(nn.ModuleList(
                [ConvNeXtBlock(in_channels=channels[i], norm=norm, padding=padding, expansion=expansion)
                 for _ in range(num_blocks[i-1])]
            ))
            self.upsampling.append(Upsampling(
                class_name=upsampling,
                in_channels=channels[i - 1],
                out_channels=channels[i],
                kernel_size=(1, 2, 2),
                padding_mode=padding,
                scale_factor=(1, 2, 2)))

    def forward(self, x, encoder_features):
        for i, stage in enumerate(self.stages):
            x = self.upsampling[i](x)
            x = x + encoder_features[i]
            for block in stage:
                x = block(x)
        return x


class UNeXt3D(nn.Module):
    # TODO: implement upsampling defaults - ResizeConv prevents checkerboard!
    def __init__(self,
                 feature_channels=(32, 64, 128, 256),
                 encoder_blocks=(1, 1, 1, 1),
                 decoder_blocks=(1, 1, 1),
                 expansion=4,
                 stem_kernel_size=(1, 5, 5),
                 stem_stride=(1, 1, 1),
                 stem_padding=(0, 2, 2),
                 norm='batch_norm',
                 padding='replicate',
                 zero_init_residual=True):

        super().__init__()
        assert len(feature_channels) == len(encoder_blocks) and len(decoder_blocks) == len(feature_channels) - 1

        # backbone
        encoder_channels = feature_channels
        decoder_channels = feature_channels[::-1]
        self.out_channels = decoder_channels[-1]

        self.encoder = Encoder(channels=encoder_channels,
                               num_blocks=encoder_blocks,
                               expansion=expansion,
                               stem_kernel_size=stem_kernel_size,
                               stem_stride=stem_stride,
                               stem_padding=stem_padding,
                               norm=norm,
                               padding=padding)
        self.decoder = Decoder(channels=decoder_channels,
                               num_blocks=decoder_blocks,
                               expansion=expansion,
                               norm=norm,
                               upsampling='ResizeConv3d',
                               padding=padding)

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
                if isinstance(m, ConvNeXtBlock):
                    nn.init.constant_(m.norm.weight, 0)
                elif isinstance(m, Downsampling):
                    nn.init.constant_(m.norm.weight, 0)

    def forward(self, x):

        # calculate encoder features
        enc_ftrs = self.encoder(x)

        # calculate decoder features
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        return out


if __name__ == "__main__":
    print("creating network ...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = rand(1, 1, 20, 128, 128).to(device)
    net = UNeXt3D(feature_channels=[8, 16, 32],
                  encoder_blocks=[1, 1, 1],
                  decoder_blocks=[1, 1],
                  expansion=4,
                  stem_kernel_size=(1, 5, 5),
                  stem_stride=(1, 1, 1),
                  norm='batch_norm',
                  padding='zeros').to(device)

    print("... network created")
    print('number of trainable parameters', sum(p.numel() for p in net.parameters() if p.requires_grad))
    start = time.time()
    print(net)
    net(t)
    print('time for forward pass', time.time() - start)
    print("Success")
