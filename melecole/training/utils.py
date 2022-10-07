import torch

from data.datasets import get_dataloader
from training.validate import Validator


def get_validator(config, device, rank, writer):
    validation_loader = get_dataloader(config,
                                       mode='val', split='val',
                                       device=device,
                                       return_view_loader=False,
                                       rank=rank,
                                       world_size=1)  # only executed on rank 0
    return Validator(validation_loader, writer, device, config)


def get_optimizer_scheduler(config, parameters):
    lr = config['OPTIM']['INITIAL_LR']
    optim = config['OPTIM']['OPTIMIZER']
    wd = config['OPTIM']['WEIGHT_DECAY']
    lr_schedule = config['OPTIM']['LR_SCHEDULER_NAME']

    if optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=wd)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    elif optim == 'AMSGrad':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=wd, amsgrad=True)
    else:
        raise NotImplementedError(f'Optimizer {optim} not implemented')

    if lr_schedule == 'CosineAnnealingWarmRestarts':
        assert config['OPTIM']['T_0'] and config['OPTIM']['T_MULT'], \
            "Please pass 'T_0', 'T_MULT' and 'RESTARTS' to 'OPTIM' config"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=config['OPTIM']['T_0'],
                                                                         T_mult=config['OPTIM']['T_MULT'],
                                                                         eta_min=0)
    elif lr_schedule == 'MultiStepLR':
        assert config['OPTIM']['MILESTONES'] and config['OPTIM']['GAMMA'], \
            "Please pass 'MILESTONES' and 'GAMMA' to 'OPTIM' config"
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['OPTIM']['MILESTONES'],
                                                         gamma=config['OPTIM']['GAMMA'])

    else:
        raise NotImplementedError(f'Scheduler {lr_schedule} not implemented')

    return optimizer, scheduler
