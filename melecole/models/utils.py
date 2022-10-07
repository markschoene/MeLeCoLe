import torch.nn as nn


def get_norm(norm, num_channels):
    return {'batch_norm': nn.BatchNorm3d(num_channels),
            'instance_norm': nn.InstanceNorm3d(num_channels)}[norm]
