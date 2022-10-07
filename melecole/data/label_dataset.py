from connectomics.data import VolumeDataset
import connectomics.data.augmentation as aug
from torch.utils.data import Dataset
from cc3d import connected_components
import numpy as np


def get_augmentor(opt):
    print(opt['AUGMENTATION'])
    kwargs = {'additional_targets': {'label': 'mask'}}
    transform = aug.Compose([
        aug.Grayscale(p=0.5,
                      contrast_factor=opt['AUGMENTATION']['CONTRAST_FACTOR'],
                      brightness_factor=opt['AUGMENTATION']['BRIGHTNESS_FACTOR'], **kwargs),
        aug.Elastic(p=0.5,
                    alpha=opt['AUGMENTATION']['WARP_ALPHA'],
                    sigma=opt['AUGMENTATION']['WARP_SIGMA'], **kwargs),
        aug.MotionBlur(p=0.5,
                       sections=opt['AUGMENTATION']['BLUR_SECTIONS'],
                       kernel_size=opt['AUGMENTATION']['BLUR_KERNEL'], **kwargs),
        aug.CutBlur(p=0.5,
                    length_ratio=opt['AUGMENTATION']['CUT_RATIO'],
                    down_ratio_min=opt['AUGMENTATION']['CUT_DOWNSAMPLE'],
                    down_ratio_max=opt['AUGMENTATION']['CUT_DOWNSAMPLE'],
                    downsample_z=False, **kwargs),
        aug.Rotate(p=0.5,
                   rot90=True, **kwargs),
        aug.Flip(p=0.5, **kwargs),
        aug.MissingParts(p=0.2,
                         iterations=opt['AUGMENTATION']['MISSING_PARTS'], **kwargs),
        aug.MissingSection(p=0.2,
                           num_sections=opt['AUGMENTATION']['MISSING_SECTIONS'], **kwargs),
        aug.MisAlignment(p=0.1,
                         displacement=opt['AUGMENTATION']['MISALIGNMENT'],
                         rotate_ratio=0, **kwargs),
    ],
        input_size=opt['MODEL']['INPUT_SIZE'],
        smooth=False,
        **kwargs)
    return transform


class SNEMI3D(Dataset):
    def __init__(self, config, mode, volumes, labels):

        # define data augmentation and corresponding crop size
        augmentor = get_augmentor(config) if mode == 'train' else None
        augmentor_size = augmentor.sample_size if mode == 'train' else config['MODEL']['INPUT_SIZE']

        sample_stride = (2, 2, 2) if mode == 'train' else config['INFERENCE']['STRIDE']
        self.recompute_cc = config['LOSS']['RECOMPUTE_CONNECTED_COMPONENTS']
        self.label_smoothing = config['DATASET']['LABEL_SMOOTHING']
        self.dataset = VolumeDataset(volume=volumes,
                                     label=labels,
                                     sample_volume_size=augmentor_size,
                                     sample_label_size=augmentor_size,
                                     sample_stride=sample_stride,
                                     augmentor=augmentor,
                                     mode=mode,
                                     target_opt=['9', '0'],
                                     erosion_rates=config['DATASET']['BOUNDARY_EROSION'],
                                     data_mean=config['DATASET']['MEAN'] / 255.,
                                     data_std=config['DATASET']['STD'] / 255.)
        self.volume_size = self.dataset.volume_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset.mode == 'train':
            pos, out_volume, out_target, out_weight = self.dataset.__getitem__(index)
            seg, foreground = out_target

            # locally recompute connected components
            if self.recompute_cc:
                mask = seg == 0
                cc = connected_components(seg, connectivity=6)
                cc[mask] = 0
                seg = np.expand_dims(cc, 0).astype(np.int64)

            # compute boundaries
            foreground = np.clip(foreground,
                                 a_min=self.label_smoothing,
                                 a_max=1 - self.label_smoothing)

            # combine segmentation ground truth and membrane ground truth to output target
            out_target = [seg, foreground]
            return pos, out_volume, out_target, out_weight
        else:
            return self.dataset.__getitem__(index)