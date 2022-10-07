import GPUtil
from torch.profiler import profile
from training.trainer import *
from training.utils import get_validator
from models import get_backbone
from training.tensorboard import Writer


def get_training_params(config, rank=0, world_size=1, distributed=False):
    assert config['OPTIM']['SUPERVISED_STEPS'] >= 0
    if 'SSL' not in config.keys():
        encoder_only = False
    elif 'DECODER' in config['SSL'].keys() and config['SSL']['DECODER']:
        encoder_only = False
    else:
        encoder_only = True

    writer = Writer(config) if rank == 0 else None

    device = torch.device(rank) if torch.cuda.is_available() else torch.device('cpu')
    backbone = get_backbone(config)

    if rank == 0 and (config['LOG']['SUPERVISED_VALIDATION'] > 0 or config['LOG']['UNSUPERVISED_VALIDATION'] > 0):
        validator = get_validator(config=config,
                                  device=device,
                                  rank=rank,
                                  writer=writer)
    else:
        validator = None

    if not config['OPTIM']['SUPERVISED_STEPS'] or config['OPTIM']['SUPERVISED_STEPS'] == 0:
        supervised_trainer = None

    else:
        supervised_trainer = SupervisedTrainer(config=config,
                                               backbone=backbone,
                                               writer=writer,
                                               device=device,
                                               rank=rank,
                                               world_size=world_size,
                                               distributed=distributed)

    if 'SSL' not in config.keys() or config['OPTIM']['UNSUPERVISED_STEPS'] == 0:
        ssl_trainer = None

    elif config['SSL']['MODEL'] == 'SimSiam':
        ssl_trainer = SimSiamTrainer(config=config,
                                     backbone=backbone,
                                     encoder_only=encoder_only,
                                     writer=writer,
                                     device=device,
                                     rank=rank,
                                     world_size=world_size,
                                     distributed=distributed)

    elif config['SSL']['MODEL'] == 'MoCo':
        ssl_trainer = MoCoTrainer(config=config,
                                  backbone=backbone,
                                  encoder_only=encoder_only,
                                  writer=writer,
                                  device=device,
                                  rank=rank,
                                  world_size=world_size,
                                  distributed=distributed)
    else:
        raise NotImplementedError(f"SSL Model {config['SSL']['MODEL']}")

    kwargs = {
        'supervised_trainer': supervised_trainer,
        'supervised_steps': config['OPTIM']['SUPERVISED_STEPS'],
        'ssl_trainer': ssl_trainer,
        'ssl_steps': config['OPTIM']['UNSUPERVISED_STEPS'],
        'validator': validator,
        'num_iterations': config['OPTIM']['TOTAL_ITER'],
        'checkpoint_frequency': config['LOG']['CHECKPOINT_FREQUENCY'],
        'writer': writer}

    return kwargs


def run_training(rank,
                 supervised_trainer,
                 supervised_steps,
                 ssl_trainer,
                 ssl_steps,
                 validator,
                 num_iterations,
                 writer,
                 checkpoint_frequency=4000,
                 profiler=None
                 ):
    assert ssl_steps >= 0 and (ssl_steps == 0 or ssl_trainer), \
        "If no self-supervised trainer is specified, cannot do ssl_steps"
    assert supervised_steps >= 0 and (supervised_steps == 0 or supervised_trainer), \
        "If no supervised trainer is specified, cannot do supervised_steps"

    for i in range(num_iterations + 1):
        if ssl_steps == 0 or i % (ssl_steps + supervised_steps) < supervised_steps:
            supervised_trainer.step(i)

            # run validation
            if rank == 0 and validator and supervised_trainer.step_count % validator.supervised_validation_frequency == 1:
                validator.validate(i, supervised_trainer.model)
        else:
            ssl_trainer.step(i)

        if i > 0 and supervised_trainer:
            supervised_trainer.scheduler.step()
        if i > 0 and ssl_trainer:
            ssl_trainer.scheduler.step()

        # monitoring
        if isinstance(profiler, profile):
            print(f'profiling iteration {i}')
            profiler.step()

        if rank == 0 and i == 0:
            GPUtil.showUtilization(all=True)

        if rank == 0 and supervised_steps > 0:
            if i % checkpoint_frequency == 0:
                writer.save_model(i, supervised_trainer.model, 'sup')
            if supervised_trainer.scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts':
                if supervised_trainer.scheduler.T_cur == supervised_trainer.scheduler.T_i - 1:
                    writer.save_model(i, supervised_trainer.model, 'recommended_sup')

        if rank == 0 and ssl_steps > 0:
            if i % checkpoint_frequency == 0:
                writer.save_model(i, ssl_trainer.model, 'ssl')
            if ssl_trainer.scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts':
                if ssl_trainer.scheduler.T_cur == ssl_trainer.scheduler.T_i - 1:
                    writer.save_model(i, ssl_trainer.model, 'recommended_ssl')

    if rank == 0:
        if supervised_steps > 0:
            writer.save_model(num_iterations, supervised_trainer.model, 'stop_sup')
        if ssl_steps > 0:
            writer.save_model(num_iterations, ssl_trainer.model, 'stop_ssl')
