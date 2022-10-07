import argparse
import os
import numpy as np
import torch.cuda

from config import Config
from training.training import run_training, get_training_params
import torch.multiprocessing as mp
import torch.distributed as dist


def get_args():
    parser = argparse.ArgumentParser(description="Training Models")
    parser.add_argument('--config', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--supervised_checkpoint', type=str, default=None,
                        help='path to load supervised the checkpoint')
    parser.add_argument('--ssl_checkpoint', type=str, default=None,
                        help='path to load the ssl checkpoint')
    return parser.parse_args()


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'

    # use the process id as a seed to a generator for port only
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train(rank, config, world_size, port):
    setup(rank=rank, world_size=world_size, port=port)

    params = get_training_params(config=config,
                                 rank=rank,
                                 world_size=world_size,
                                 distributed=True)
    run_training(rank, **params)

    cleanup()


if __name__ == '__main__':
    args = get_args()
    cfg = Config(args.config)
    cfg.config['supervised_checkpoint'] = args.supervised_checkpoint
    cfg.config['ssl_checkpoint'] = args.ssl_checkpoint
    cfg.config['tag'] = args.config

    config = cfg.config
    world_size = config['HARDWARE']['NUM_GPUS']

    # choose a random master-port
    pid = os.getpid()
    rng1 = np.random.RandomState(pid)
    port = rng1.randint(10000, 19999, 1)[0]

    mp.spawn(train,
             args=(config, world_size, port),
             nprocs=world_size)
