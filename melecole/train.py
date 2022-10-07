import argparse
from config import Config
from training.training import run_training, get_training_params


def get_args():
    parser = argparse.ArgumentParser(description="Training Models")
    parser.add_argument('--config', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--supervised_checkpoint', type=str, default=None,
                        help='path to load a checkpoint from a supervised model')
    parser.add_argument('--ssl_checkpoint', type=str, default=None,
                        help='path to load the ssl checkpoint')
    return parser.parse_args()


def train(config):
    params = get_training_params(config=config)
    run_training(rank=0, **params)
    print('TRAINING FINISHED')


if __name__ == '__main__':
    args = get_args()
    cfg = Config(args.config)
    cfg.config['supervised_checkpoint'] = args.supervised_checkpoint
    cfg.config['ssl_checkpoint'] = args.ssl_checkpoint
    cfg.config['tag'] = args.config

    train(config=cfg.config)
