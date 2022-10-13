# MeLeCoLe - Metric Learning and Contrastive Learning for Neuron Segmentation
MeLeCoLe is a neuron segmentation library written in pytorch. 
It reproduces the metric learning approach proposed by [Kisuk Lee et al.](https://arxiv.org/abs/1909.09872) and adds a [momentum contrastive](https://arxiv.org/abs/1911.05722) training branch to make representations more robust and sample efficient.

This library makes heavy use of the [MoCo Repository](https://github.com/facebookresearch/moco/tree/main/moco) and the [Pytorch Connectomics](https://connectomics.readthedocs.io/en/latest/) library.

Contrastive learning with 3D convolutions is computationally very expensive.
I did not have the computational budget available to find a working configuration.
So, the contrastive learning functionality should be treated experimentally!

## Requirements

- [cc3d](https://github.com/seung-lab/connected-components-3d)
- [MutexWatershed](https://github.com/hci-unihd/mutex-watershed), which requires [xtl](https://github.com/xtensor-stack/xtl) and [xtensor](https://github.com/xtensor-stack/xtensor)
- [PytorchConnectomics](https://connectomics.readthedocs.io/en/latest/notes/installation.html#)
- [partition-comparison](https://github.com/thouis/partition-comparison)
- [cloud-volume](https://github.com/seung-lab/cloud-volume)

## Get the data
When working with SNEMI, please cite the following original publication
Kasthuri, Narayanan, et al. "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661. https://doi.org/10.1016/j.cell.2015.06.054

The training, validation and test split used in this work can be downloaded in HDF5 format from
[TU Dresden's cloud storage](https://cloudstore.zih.tu-dresden.de/index.php/s/DeWzTtfK7bQWea5)

## Training
The library offers functions for single- and multi GPU training for both supervised metric learning as well as momentum contrastive learning.

### Supervised Single GPU Training
To train from scratch on a single GPU run

    python melecole/train.py --config configs/SNEMI3D-UNet3D.yaml

To train starting from a checkpointed MeLeCoLe model run

    python melecole/train.py --config configs/SNEMI3D-UNet3D.yaml --supervised_checkpoint PATH_TO_MODEL

Any custom config can be used to train models.
However, to continue training from a checkpoint the parameters for model initialization must match.
Such parameters contain the filter size, input size, number of layers etc.
Model independent parameters such as learning rate, weight decay etc can be adjusted even for training from a checkpointed model.

### Distributed Multi-GPU Training
MeLeCoLe supports multi-GPU training to train with larger batch sizes.
To run multi-GPU training adjust the config to feature more than one GPU.
E.g. to train on 4 GPUs do

    HARDWARE:
        NUM_GPUS: 4

And run the distributed training command

    python melecole/train_distributed.py --config configs/SNEMI3D-UNet3D.yaml

### Momentum Contrastive Learning
MeLeCoLe implements momentum contrastive learning closely following [He et al.'s](https://github.com/facebookresearch/moco/blob/main/moco/builder.py) implementation.

TODO: Provide configs

## Inference
Inference can be run on any dataset following the data convention used in this repository by

    python infer.py --split val --config PATH_TO_CONFIG --checkpoint PATH_TO_MODEL

We obviously need to run inference with a trained model. 
The results will be saved as a subdirectory in the same directory where the trained model was logged.
Membranes, affinities, and segmentations are saved.

## Evaluation
Quantitative evaluation is conducted by means of variation of information via the [partition-comparison](https://github.com/thouis/partition-comparison) package.

    python variation_of_information.py --gt dataset/val/AC3_val.hdf --segmentation <path-to-segmentations.hdf> --membranes <path-to-membranes.hdf>

View your segmentation results via

    python -i view.py --gt dataset/val/AC3_val.hdf --segmentation <path-to-segmentations.hdf> --padding 20 64 64# MeLeCoLe
