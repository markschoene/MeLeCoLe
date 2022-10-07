from __future__ import print_function, division
from typing import Optional, List
import numpy as np
import random
import torch.utils.data
from connectomics.data.augmentation import Compose, DataAugment
import connectomics.data.augmentation as aug
from connectomics.data.utils import *

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]


def get_view_augmentor(input_size,
                       crop_displacement,
                       contrast_factor=0.1,
                       brightness_factor=0.1,
                       warp_alpha=8,
                       warp_sigma=4,
                       motion_blur_size=5,
                       rescale_low=1,
                       rescale_high=1.25):
    transform = aug.Compose([
        aug.Rescale(low=rescale_low, high=rescale_high, fix_aspect=False, p=0.5),
        aug.Grayscale(contrast_factor=contrast_factor, brightness_factor=brightness_factor, p=0.75),
        aug.Elastic(alpha=warp_alpha, sigma=warp_sigma, p=0.5),
        aug.MotionBlur(sections=4, kernel_size=motion_blur_size, p=0.5),
        RandomCrop(crop_size=input_size, displacement=crop_displacement, p=1)
        # aug.CutBlur(length_ratio=0.25, p=0.5, down_ratio_min=2.0, down_ratio_max=2.0, downsample_z=False),
        # aug.CutNoise(length_ratio=0.25, p=0.5, scale=0.1),
        # aug.MissingParts(p=0.2, iterations=32),
        # aug.MissingSection(p=0.2),
        # aug.MisAlignment(displacement=16, p=0.2, rotate_ratio=0),
    ],
        input_size=input_size,
        smooth=True)
    return transform


def crop_volume(data, sz, st=(0, 0, 0)):  # C*D*W*H, C=1
    st = np.array(st).astype(np.int32)
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]


def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)


def normalize_image(image: np.ndarray,
                    mean: float = 0.5,
                    std: float = 0.5) -> np.ndarray:
    assert image.dtype == np.float32
    image = (image - mean) / std
    return image


class ViewVolumeDataset(torch.utils.data.Dataset):
    """
    Dataset class for volumetric image datasets. At training time, subvolumes are randomly sampled from all the large
    input volumes with (optional) rejection sampling to increase the frequency of foreground regions in a batch. At inference
    time, subvolumes are yielded in a sliding-window manner with overlap to counter border artifacts.

    Args:
        volume (list): list of image volumes.
        label (list, optional): list of label volumes. Default: None
        valid_mask (list, optional): list of valid masks. Default: None
        valid_ratio (float): volume ratio threshold for valid samples. Default: 0.5
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor (connectomics.data.augmentation.composition.Compose, optional): data augmentor for training. Default: None
        target_opt (list): list the model targets generated from segmentation labels.
        weight_opt (list): list of options for generating pixel-wise weight masks.
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        do_2d (bool): load 2d samples from 3d volumes. Default: False
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        reject_size_thres (int, optional): threshold to decide if a sampled volumes contains foreground objects. Default: 0
        reject_diversity (int, optional): threshold to decide if a sampled volumes contains multiple objects. Default: 0
        reject_p (float, optional): probability of rejecting non-foreground volumes. Default: 0.95

    Note:
        For relatively small volumes, the total number of possible subvolumes can be smaller than the total number
        of samples required in training (the product of total iterations and mini-natch size), which raises *StopIteration*.
        Therefore the dataset length is also decided by the training settings.
    """

    background: int = 0  # background label index

    def __init__(self,
                 volume: list,
                 augmentor: AUGMENTOR_TYPE,
                 sample_volume_size: tuple,
                 sample_stride: tuple = (1, 1, 1),
                 iter_num: int = -1,
                 # normalization
                 data_mean: float = 0.5,
                 data_std: float = 0.5):

        # data format
        self.volume = volume

        # augmentation
        self.augmentor = augmentor

        # normalization
        self.data_mean = data_mean
        self.data_std = data_std

        # dataset: channels, depths, rows, cols
        # volume size, could be multi-volume input
        self.volume_size = [np.array(x.shape) for x in self.volume]
        self.sample_volume_size = np.array(
            sample_volume_size).astype(int)  # model input size

        self._assert_valid_shape()

        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(sample_stride).astype(int)
        self.sample_size = [count_volume(self.volume_size[x], self.sample_volume_size, self.sample_stride)
                            for x in range(len(self.volume_size))]
        self.sample_size_test = [
            np.array([np.prod(x[1:3]), x[2]]) for x in self.sample_size]

        # total number of possible inputs for each volume
        self.sample_num = np.array([np.prod(x) for x in self.sample_size])
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training.
        self.iter_num = max(iter_num, self.sample_num_a)
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32

        # crop a volume
        vol_size = self.sample_volume_size
        pos = self._get_pos(vol_size)
        out_volume = (crop_volume(self.volume[pos[0]], vol_size, pos[1:]) / 255.0).astype(np.float32)
        # generate views

        data_a = {'image': out_volume}
        data_b = {'image': out_volume.copy()}

        view_a = self.augmentor(data_a)
        view_b = self.augmentor(data_b)

        image_a = normalize_image(np.expand_dims(view_a['image'], 0),
                                  mean=self.data_mean,
                                  std=self.data_std)
        image_b = normalize_image(np.expand_dims(view_b['image'], 0),
                                  mean=self.data_mean,
                                  std=self.data_std)

        return pos, image_a, image_b

    #######################################################
    # Position Calculator
    #######################################################

    def _index_to_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def _index_to_location(self, index, sz):
        # index -> z,y,x
        # sz: [y*x, x]
        pos = [0, 0, 0]
        pos[0] = np.floor(index/sz[0])
        pz_r = index % sz[0]
        pos[1] = int(np.floor(pz_r/sz[1]))
        pos[2] = pz_r % sz[1]
        return pos

    def _get_pos(self, vol_size):
        # random: multithread
        # np.random: same seed
        pos = [0, 0, 0, 0]
        # pick a dataset
        did = self._index_to_dataset(random.randint(0, self.sample_num_a - 1))
        pos[0] = did
        # pick a position
        tmp_size = count_volume(
            self.volume_size[did], vol_size, self.sample_stride)
        tmp_pos = [random.randint(0, tmp_size[x] - 1) * self.sample_stride[x]
                   for x in range(len(tmp_size))]

        pos[1:] = tmp_pos
        return pos

    #######################################################
    # Utils
    #######################################################
    def _assert_valid_shape(self):
        assert all(
            [(self.sample_volume_size <= x).all()
             for x in self.volume_size]
        ), "Input size should be smaller than volume size."


class RandomCrop(DataAugment):
    r"""
    Rescale augmentation. This augmentation is applied to both images and masks.

    Args:
        crop_size (tuple): size to crop the sample
        displacement (tuple): maximum shift away from the center
        p (float): probability of applying the augmentation. Default: 0.5
    """

    def __init__(self,
                 crop_size: tuple,
                 displacement: tuple = (4, 32, 32),
                 p: float = 0.5,):

        super(RandomCrop, self).__init__(p)
        self.crop_size = np.array(crop_size)
        self.displacement = np.array(displacement)

        self.set_params()

    def set_params(self):
        r"""The rescale augmentation is only applied to the `xy`-plane. The required
        sample size before transformation need to be larger as decided by the lowest
        scaling factor (:attr:`self.low`).
        """
        self.sample_params['add'] = self.displacement
        self.sample_params['ratio'] = [1.0, 1.0, 1.0]

    def get_random_params(self, images, random_state):
        margin = (images.shape - self.crop_size) // 2
        displacement = np.minimum(margin, self.displacement)
        z0, y0, x0 = margin + random_state.randint(-displacement, displacement)
        assert z0 >= 0 and y0 >= 0 and x0 >= 0

        return z0, y0, x0

    def apply_random_crop(self, image, z0, y0, x0):
        z1, y1, x1 = self.crop_size
        image = image[z0:z0+z1, y0:y0+y1, x0:x0+x1]

        return image

    def __call__(self, sample, random_state=np.random.RandomState()):
        if random_state.rand() < self.p:
            images = sample['image'].copy()
            z0, y0, x0 = self.get_random_params(images, random_state)
            sample['image'] = self.apply_random_crop(images, z0, y0, x0)
        return sample
