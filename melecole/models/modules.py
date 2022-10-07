import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import get_norm


class MultiLayerEmbeddingHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm, padding_mode, scale_factor, kernel_size=(1, 3, 3)):
        super(MultiLayerEmbeddingHead, self).__init__()

        self.scale_factor = nn.Parameter(torch.Tensor([scale_factor]), requires_grad=True)
        self.head = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding='same', padding_mode=padding_mode),
            get_norm(norm, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding='same', padding_mode=padding_mode),
            get_norm(norm, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1, 1, 1), padding='same', padding_mode=padding_mode)
        )

    def forward(self, x):
        return self.scale_factor * self.head(x)


class EmbeddingHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm, padding_mode, scale_factor, kernel_size=(1, 5, 5)):
        super(EmbeddingHead, self).__init__()

        self.scale_factor = nn.Parameter(torch.Tensor([scale_factor]), requires_grad=True)
        self.head = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding='same', padding_mode=padding_mode))

    def forward(self, x):
        return self.scale_factor * self.head(x)


class MultiLayerBoundaryHead(nn.Module):
    def __init__(self, in_channels, norm, padding_mode, kernel_size=(1, 5, 5)):
        super(MultiLayerBoundaryHead, self).__init__()

        self.head = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding='same', padding_mode=padding_mode),
            get_norm(norm, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=1,
                      kernel_size=(1, 1, 1), padding='same', padding_mode=padding_mode)
        )

    def forward(self, x):
        return self.head(x)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels, norm, padding_mode, kernel_size=(1, 5, 5)):
        super(BoundaryHead, self).__init__()

        self.head = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=1,
                      kernel_size=kernel_size, padding='same', padding_mode=padding_mode))

    def forward(self, x):
        return self.head(x)


class MLPProjectionHead(nn.Module):
    def __init__(self, prev_dim, dim, kernel_size):
        super(MLPProjectionHead, self).__init__()

        self.pool = nn.AvgPool3d(kernel_size=kernel_size)

        self.head = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim),
                                  nn.ReLU(inplace=True),  # first layer
                                  nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim),
                                  nn.ReLU(inplace=True),  # second layer
                                  nn.Linear(prev_dim, dim),
                                  nn.BatchNorm1d(dim, affine=False))  # output layer
        self.head[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

    def forward(self, x):
        assert len(x.shape) == 5, 'input must be (N, C, D, H, W)'
        x = self.pool(x)
        N, C, D, H, W = x.shape
        x = x.view(-1, C)
        x = self.head(x)
        return x


class MoCoHead(nn.Module):
    def __init__(self, prev_dim, dim, kernel_size):
        super(MoCoHead, self).__init__()

        self.pool = nn.AvgPool3d(kernel_size=kernel_size)

        self.head = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim),
                                  nn.ReLU(inplace=True),  # first layer
                                  nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim),
                                  nn.ReLU(inplace=True),  # second layer
                                  nn.Linear(prev_dim, dim),
                                  nn.BatchNorm1d(dim, affine=False))  # output layer
        self.head[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

    def forward(self, x):
        assert len(x.shape) == 5, 'input must be (N, C, D, H, W)'
        x = self.pool(x)
        N, C, D, H, W = x.shape
        x = x.view(-1, C)
        x = self.head(x)
        return x.view(N, -1, C)


class PatchProjectionHead(nn.Module):
    def __init__(self, prev_dim, dim, kernel_size=(2, 4, 4), expansion=4):
        super(PatchProjectionHead, self).__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm3d(prev_dim),
            nn.Conv3d(in_channels=prev_dim, out_channels=int(prev_dim * expansion),
                      kernel_size=kernel_size, stride=kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(int(prev_dim * expansion)),
            nn.Conv3d(in_channels=int(prev_dim * expansion), out_channels=int(prev_dim * expansion * expansion),
                      kernel_size=kernel_size, stride=kernel_size),
            nn.ReLU()
        )
        self.head = nn.Linear(int(prev_dim * expansion * expansion), dim)

    def forward(self, x):
        assert len(x.shape) == 5, 'input must be (N, C, D, H, W)'
        x = self.encoder(x)
        N, C, D, H, W = x.shape
        x = nn.functional.avg_pool3d(x, kernel_size=(D, H, W))
        x = x.view(N, C)
        x = self.head(x)
        return x


class ResizeConv3d(nn.Module):
    """
    3D Resize convolution to avoid checkerboard artifacts
    Reference: https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode, scale_factor):
        super(ResizeConv3d, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding='same', padding_mode=padding_mode)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(self,
                 class_name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding_mode,
                 scale_factor=(1, 2, 2)):

        super().__init__()

        if class_name == 'ConvTranspose3d':
            self.m = nn.ConvTranspose3d(in_channels, out_channels,
                                        kernel_size=kernel_size, stride=scale_factor)

        elif class_name == 'ResizeConv3d':
            self.m = ResizeConv3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  padding_mode=padding_mode,
                                  scale_factor=scale_factor)

        else:
            raise NotImplementedError(f'Upsampling class {class_name} not implemented')

    def forward(self, x):
        return self.m(x)


class Downsampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm,
                 stride=(1, 2, 2),
                 pool=False):

        super().__init__()

        self.norm = get_norm(norm, in_channels)
        self.activation = nn.ReLU() if pool else nn.Identity()
        self.pool = nn.MaxPool3d(kernel_size=stride, stride=stride) if pool else nn.Identity()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.pool(F.relu(self.norm(x)))
        return self.conv(x)
