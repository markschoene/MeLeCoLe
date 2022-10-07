import h5py
import os.path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.label_dataset import SNEMI3D
from data.view_dataset import get_view_augmentor, ViewVolumeDataset


def get_dataset(config, mode, split):
    """
    Loads the VolumeDataset depending on mode (whether to train and augment or to infere and pad)
    and dataset split (train, val, test)

    :param config: (dict) Config dictionary
    :param mode: (str) Either 'train' or 'test' -> sets the data loading mode
    :param split: (str) 'train', 'val', or 'test' -> Loads the relevant dataset split
    :return: VolumeDataset
    """
    root = config['DATASET']['DATA_ROOT']
    dataset_name = config['DATASET']['NAME']
    padding = config['DATASET']['PADDING']

    directory = os.path.join(root, split)
    assert os.path.isdir(directory), NotADirectoryError(directory)

    if dataset_name == 'CREMI':
        volumes = [h5py.File(os.path.join(directory, f), 'r')['volume'] for f in os.listdir(directory)]
        labels = [h5py.File(os.path.join(directory, f), 'r')['labels'] for f in os.listdir(directory)]

    elif dataset_name == 'SNEMI3D':
        volumes = [h5py.File(os.path.join(directory, f), 'r')['volume'][:]
                   for f in os.listdir(directory)]
        if mode == 'train' or mode == 'val':
            volumes = [v[padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2]]
                       for v in volumes]
        labels = [h5py.File(os.path.join(directory, f), 'r')['labels'][:] for f in os.listdir(directory)]

    else:
        raise NotImplementedError

    assert volumes and labels, "No Volumes or Labels loaded"

    print('Dataset shapes:', [v.shape for v in volumes])

    return SNEMI3D(config=config,
                   mode=mode,
                   volumes=volumes,
                   labels=labels)


def get_view_dataset(config):
    root = config['DATASET']['UNLABELED_DATA_ROOT']
    dataset_name = config['DATASET']['NAME']

    assert os.path.isdir(root), NotADirectoryError(root)

    if dataset_name == 'CREMI':
        volumes = [h5py.File(os.path.join(root, f), 'r')['volume'][:]
                   for f in os.listdir(root)]

    elif dataset_name == 'SNEMI3D':
        volumes = [h5py.File(os.path.join(root, f), 'r')['volume'][:]
                   for f in os.listdir(root)]

    else:
        raise NotImplementedError

    assert volumes, "No Volumes or Labels loaded"

    print('Dataset shapes:', [v.shape for v in volumes])

    augmentor_size = config['DATASET']['SSL_CROP_SIZE']
    augmentor = get_view_augmentor(input_size=config['MODEL']['INPUT_SIZE'],
                                   crop_displacement=config['AUGMENTATION']['CROP_DISPLACEMENT'],
                                   rescale_low=config['AUGMENTATION']['RESCALE_LOW'],
                                   rescale_high=config['AUGMENTATION']['RESCALE_HIGH'],
                                   contrast_factor=0.1,
                                   brightness_factor=0.1,
                                   warp_alpha=8,
                                   warp_sigma=4,
                                   motion_blur_size=5)
    return ViewVolumeDataset(volume=volumes,
                             augmentor=augmentor,
                             sample_volume_size=augmentor_size,
                             sample_stride=(1, 1, 1),
                             data_mean=config['DATASET']['MEAN'] / 255.,
                             data_std=config['DATASET']['STD'] / 255.)


def get_distributed_sampler(dataset, rank, world_size):
    if world_size > 1:
        return DistributedSampler(dataset,
                                  num_replicas=world_size,
                                  rank=rank,
                                  shuffle=False,
                                  drop_last=True)
    else:
        return None


def get_dataloader(config, mode, split, device, return_view_loader=False, rank=0, world_size=1):
    batch_size = config['OPTIM']['BATCH_SIZE'] if mode == 'train' else config['INFERENCE']['BATCH_SIZE']
    num_workers = {'train': config['HARDWARE']['NUM_WORKERS'],
                   'val': config['INFERENCE']['THREADS'],
                   'test': config['INFERENCE']['THREADS']}

    if return_view_loader:
        dataset = get_view_dataset(config)
    else:
        dataset = get_dataset(config, mode, split)

    # the supervised samples data loader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers[mode],
                            shuffle=False,
                            sampler=get_distributed_sampler(dataset=dataset,
                                                            rank=rank,
                                                            world_size=world_size),
                            pin_memory=True if device.type == 'cuda' else False,
                            drop_last=True)
    return dataloader
