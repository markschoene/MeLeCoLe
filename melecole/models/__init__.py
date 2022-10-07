import torch.nn as nn
from torch import sigmoid
from collections import namedtuple
import models.modules as heads
from models.unet import UNet
from models.unext import UNeXt3D

OutputTuple = namedtuple('OutputTuple', 'embeddings boundaries')
ProjectionTuple = namedtuple('ProjectionTuple', 'embeddings projection')


class SupervisedModel(nn.Module):
    def __init__(self, backbone, embedding_head, boundary_head, embedding_dimension, norm, padding_mode):
        super(SupervisedModel, self).__init__()
        self.backbone = backbone
        self.embedding_head = getattr(heads, embedding_head)(
            in_channels=backbone.out_channels,
            out_channels=embedding_dimension,
            norm=norm,
            padding_mode=padding_mode,
            scale_factor=0.1)
        self.boundary_head = getattr(heads, boundary_head)(
            in_channels=backbone.out_channels,
            norm=norm,
            padding_mode=padding_mode)
        print(self)

    def forward(self, x):
        x = self.backbone(x)
        if self.training:
            return OutputTuple(self.embedding_head(x), self.boundary_head(x))
        else:
            return OutputTuple(self.embedding_head(x), sigmoid(self.boundary_head(x)))


class SelfSupervisedModel(nn.Module):
    def __init__(self, backbone, feature_pooling_kernel_size, projection_dim, projection_head, encoder_only=False):
        super(SelfSupervisedModel, self).__init__()
        self.backbone = backbone if not encoder_only else backbone.encoder
        self.encoder_only = encoder_only
        self.projection_head = getattr(modules, projection_head)(prev_dim=self.backbone.out_channels,
                                                                 dim=projection_dim,
                                                                 kernel_size=feature_pooling_kernel_size)
        if encoder_only:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.encoder_only:
            x = self.backbone(x)[-1]
            return ProjectionTuple(x, self.projection_head(x))
        else:
            x = self.backbone(x)
            return ProjectionTuple(x, self.projection_head(x))


class Predictor(nn.Module):
    def __init__(self, dim, pred_dim):
        super(Predictor, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x):
        return self.head(x)


def get_backbone(config):

    zero_init_residuals = False
    if 'ZERO_INIT_RESIDUALS' in config['MODEL'].keys() and config['MODEL']['ZERO_INIT_RESIDUALS']:
        zero_init_residuals = True

    if 'CLASS' not in config['MODEL'].keys() or config['MODEL']['CLASS'] == 'UNet':
        model = UNet(feature_channels=config['MODEL']['FILTERS'],
                     stem_kernel_size=config['MODEL']['STEM_KERNEL_SIZE'],
                     stem_stride=config['MODEL']['STEM_STRIDE'],
                     stem_padding=config['MODEL']['STEM_PADDING'],
                     padding=config['MODEL']['PADDING'],
                     norm=config['MODEL']['NORM'],
                     zero_init_residual=zero_init_residuals
                     )

    elif config['MODEL']['CLASS'] == 'UNeXt3D':
        model = UNeXt3D(feature_channels=config['MODEL']['FILTERS'],
                        encoder_blocks=config['MODEL']['ENCODER_BLOCKS'],
                        decoder_blocks=config['MODEL']['DECODER_BLOCKS'],
                        expansion=config['MODEL']['EXPANSION'],
                        stem_kernel_size=config['MODEL']['STEM_KERNEL_SIZE'],
                        stem_stride=config['MODEL']['STEM_STRIDE'],
                        stem_padding=config['MODEL']['STEM_PADDING'],
                        norm=config['MODEL']['NORM'],
                        padding=config['MODEL']['PADDING'],
                        zero_init_residual=zero_init_residuals
                        )

    else:
        raise NotImplementedError(f"Model class {config['MODEL']['CLASS']} not implemented")

    return model
