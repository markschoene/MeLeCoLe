import torch
import argparse
import numpy as np
from data.datasets import get_dataloader
from models import get_backbone, SupervisedModel
from config import Config
from connectomics.data.utils import build_blending_matrix, writeh5
from mutex_watershed import MutexWatershed
import os
import time
import GPUtil
import h5py


def get_args():
    parser = argparse.ArgumentParser(description="Computing predictions")
    parser.add_argument('--config', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--local_rank', type=int,
                        help='node rank for distributed training', default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to load the checkpoint')
    parser.add_argument('--split', type=str, default='test',
                        help='Loads the relevant dataset split')
    args = parser.parse_args()
    return args


def setup_inference(config, split):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    backbone = get_backbone(config)
    model = SupervisedModel(backbone=backbone,
                            embedding_head=config['MODEL']['EMBEDDING_HEAD'],
                            boundary_head=config['MODEL']['BOUNDARY_HEAD'],
                            embedding_dimension=config['MODEL']['EMBEDDING_DIMENSION'],
                            norm=config['MODEL']['NORM'],
                            padding_mode=config['MODEL']['PADDING']).to(device)

    model.load_state_dict(torch.load(config['checkpoint'], map_location=device))
    model.eval()

    dataloader = get_dataloader(config, mode='test', split=split, device=device)

    return dataloader, model, device


def get_crop_offset_pair(embeddings, crop_coordinates, offset):
    slices = [slice(s.start + offset[l], s.stop + offset[l]) for l, s in enumerate(crop_coordinates)]
    return embeddings[:, :, crop_coordinates[0], crop_coordinates[1], crop_coordinates[2]],\
           embeddings[:, :, slices[0], slices[1], slices[2]]


def get_cropped_affinity(embeddings, crop_coordinates, offset, delta):
    cropped_embeddings, offset_embeddings = get_crop_offset_pair(embeddings, crop_coordinates, offset)
    batch_size, n_dims, D, H, W = cropped_embeddings.shape
    affinity = torch.maximum((2 * delta - torch.norm(cropped_embeddings - offset_embeddings, dim=1)) / 2 / delta,
                             torch.zeros((batch_size, D, H, W), device=cropped_embeddings.device)) ** 2
    return affinity


def get_results_path(config, name):
    output_dir = os.path.dirname(os.path.dirname(config['checkpoint']))
    results_dir = os.path.join(output_dir, 'results')
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    return os.path.join(results_dir, f'{name}.hdf')


def compute_affinities(config, split, offset_vectors):
    """
    Code for inference is based on
    https://connectomics.readthedocs.io/en/latest/

    :param config: Config file as defined by configs/config.py
    :param split: (str) 'train', 'val', or 'test' -> Loads the relevant dataset split
    :param offset_vectors: (array) list of offset vecotrs for affinities
    :return: Saves an output hdf5 file
    """
    delta = config['LOSS']['DELTA']

    spatial_size = config['MODEL']['OUTPUT_SIZE']
    output_crop = config['INFERENCE']['OUTPUT_CROP_SIZE']
    volume_margin = config['DATASET']['PADDING']
    channel_size = len(offset_vectors)

    # check if affinities are already available
    affinities_path = get_results_path(config, f'{split}_affinities')
    membranes_path = get_results_path(config, f'{split}_membranes')
    if os.path.isfile(affinities_path) and os.path.isfile(membranes_path):
        print(f"READING SAVED PREDICTIONS FROM {affinities_path} and {membranes_path}")
        with h5py.File(affinities_path, "r") as f:
            affinities = [f[key][:] for key in f.keys()]
        with h5py.File(membranes_path, "r") as f:
            membranes = [f[key][:] for key in f.keys()]
        return affinities, membranes

    dataloader, model, device = setup_inference(config, split)

    sz = tuple([channel_size] + spatial_size)
    margin = (np.subtract(spatial_size, output_crop) // 2).astype(np.uint32)
    crop_slices = [slice(margin[l], margin[l] + output_crop[l]) for l in range(3)]
    ww = build_blending_matrix(output_crop, config['INFERENCE']['BLENDING'])

    output_size = [tuple(x) for x in dataloader.dataset.volume_size]
    result_affinities = [np.stack([np.zeros(x, dtype=np.float32)
                                   for _ in range(channel_size)]) for x in output_size]
    result_membranes = [np.zeros(x, dtype=np.float32)
                        for x in output_size]
    weight = [np.zeros(x, dtype=np.float32) for x in output_size]
    print("Total number of batches: ", len(dataloader))
    print(f"Affinities shape {[r.shape for r in result_affinities]}")
    print(f"Membranes shape {[r.shape for r in result_membranes]}")

    start = time.perf_counter()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(dataloader):
            print(f'progress: {i + 1:d}/{len(dataloader):d} batches, total time {(time.perf_counter() - start):.2f}')
            volume = volume.to(device, non_blocking=True)
            output = model(volume)
            embeddings = output.embeddings  # self.augmentor(self.model, volume)
            boundaries = output.boundaries

            # iterate offset vectors for dynamical affinity computation
            affinities = torch.zeros(
                (embeddings.shape[0], len(offset_vectors), output_crop[0], output_crop[1], output_crop[2]))
            for j, offset in enumerate(offset_vectors):
                affinities[:, j] = get_cropped_affinity(embeddings, crop_slices, offset, delta)

            if torch.cuda.is_available() and i == 0:
                GPUtil.showUtilization(all=True)

            # add affinities and membranes to results file
            pos = torch.transpose(torch.stack(pos), 0, 1).cpu().numpy()
            for idx in range(affinities.shape[0]):
                st = pos[idx]
                out_block = affinities[idx]
                out_bounds = boundaries[idx][0, crop_slices[0], crop_slices[1], crop_slices[2]]
                result_slices = [slice(st[l + 1] + margin[l], st[l + 1] + sz[l + 1] - margin[l]) for l in range(3)]
                result_affinities[st[0]][:, result_slices[0], result_slices[1], result_slices[2]] += (
                        out_block.cpu().numpy() * ww[np.newaxis, :])
                result_membranes[st[0]][result_slices[0], result_slices[1], result_slices[2]] += (
                        out_bounds.cpu().numpy() * ww)
                weight[st[0]][result_slices[0], result_slices[1], result_slices[2]] += ww

    end = time.perf_counter()
    print("Prediction time: %.2fs" % (end - start))

    for vol_id in range(len(result_affinities)):

        # slice off the margins of the volumes
        slicer = tuple(slice(volume_margin[i], result_affinities[vol_id].shape[i+1] - volume_margin[i])
                       for i in range(3))

        weight[vol_id] = weight[vol_id][slicer]

        # weight affinities
        result_affinities[vol_id] = result_affinities[vol_id][(slice(None), ) + slicer]
        result_affinities[vol_id] /= np.expand_dims(weight[vol_id], axis=0)
        result_affinities[vol_id] = result_affinities[vol_id].astype(np.float32)

        # weight membranes
        result_membranes[vol_id] = result_membranes[vol_id][slicer]
        result_membranes[vol_id] /= weight[vol_id]
        result_membranes[vol_id] = result_membranes[vol_id].astype(np.float32)

    print('Final prediction shapes are:')
    for aff in result_affinities:
        print(aff.shape)
    writeh5(affinities_path, result_affinities,
            ['vol%d' % (x) for x in range(len(result_affinities))])
    print('Affinities saved as: ', affinities_path)
    writeh5(membranes_path, result_membranes,
            ['vol%d' % (x) for x in range(len(result_affinities))])
    print('Membranes saved as: ', membranes_path)
    return result_affinities, result_membranes


def run_mws(affinities,
            offsets, stride,
            foreground_mask,
            seperating_channel=3,
            invert_dam_channels=True,
            bias_cut=0.,
            randomize_bounds=True,):
    """

    This code was copied from https://github.com/hci-unihd/mutex-watershed/

    """

    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')
    if invert_dam_channels:
        affinities_[seperating_channel:] *= -1
        affinities_[seperating_channel:] += 1
    affinities_[:seperating_channel] += bias_cut

    # sort in descending order
    sorted_edges = np.argsort(-affinities_.ravel())

    # remove edges adjacent to background voxels from graph
    sorted_edges = sorted_edges[foreground_mask.ravel()[sorted_edges]]

    # run the mutex watershed
    vol_shape = affinities_.shape[1:]
    mst = MutexWatershed(np.array(vol_shape),
                         offsets,
                         seperating_channel,
                         stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_ucc_mst_cut(sorted_edges, 0)
    segmentation = mst.get_flat_label_image_only_merged_pixels().reshape(vol_shape)
    return segmentation


def get_foreground_mask(affinities, membranes, offsets, theta):
    """
    Compute the foreground affinities given membrane predictions and a threshold.

    :param affinities: Metric affinities
    :param membranes: Foreground prediction (where 1 is foreground and 0 is background)
    :param offsets: offset vectors defining the affinity map
    :param theta: Threshold
    :return: Foreground mask (where 1 is foreground and 0 is background) that removes all background voxels and their corresponding affinities
    """
    mask = np.ones_like(affinities).astype(bool)

    # get largest offset number
    pad_size = np.max(np.abs(np.array(offsets)))

    # initialize padded foreground
    foregound = np.pad(membranes > theta, pad_width=pad_size, mode='constant', constant_values=1).astype(bool)

    # compute foreground mask for each offset vector
    for i, vector in enumerate(offsets):
        dims = membranes.shape
        slices_null = [slice(pad_size, pad_size + dims[k]) for k in range(len(dims))]
        slices_plus = [slice(pad_size + vector[k], pad_size + vector[k] + dims[k]) for k in range(len(dims))]

        # remove both edges that are associated with pixel (i, j, k)
        # that is (offset_1, offset_2, offset_3) + (i, j, k) AND (i, j, k)
        mask[i] = np.logical_and(foregound[slices_plus[0], slices_plus[1], slices_plus[2]],
                                 foregound[slices_null[0], slices_null[1], slices_null[2]])

    print(f"Detected {np.sum(mask):.2e} foreground affinities out of {np.prod(mask.shape):.2e} total affinities")
    return mask


def run_inference(config, split):
    offsets = np.array(config['INFERENCE']['OFFSET_VECTORS'])
    stride = np.array(config['INFERENCE']['MWS_STRIDE'])
    background_theta = config['INFERENCE']['MWS_BACKGROUND']
    assert offsets.size > 0, "No offset vectors have been set. Please set INFERENCE-OFFSET_VECTORS"

    affinities, membranes = compute_affinities(config, split=split, offset_vectors=offsets)
    segmentations = []

    for i, (aff, mem) in enumerate(zip(affinities, membranes)):
        print(f"Running Mutex Watershed for volume {i+1} / {len(affinities)}")
        segmentations.append(run_mws(affinities=aff,
                                     offsets=offsets,
                                     stride=stride,
                                     foreground_mask=get_foreground_mask(aff, mem, offsets, background_theta),
                                     seperating_channel=3,
                                     bias_cut=0,
                                     invert_dam_channels=True,
                                     randomize_bounds=False))

    results_path = get_results_path(config, f"{split}_segmentations_{str(background_theta).replace('.', '')}")
    writeh5(results_path, segmentations,
            ['vol%d' % (x) for x in range(len(segmentations))])


if __name__ == '__main__':
    args = get_args()
    cfg = Config(args.config)
    cfg.config['checkpoint'] = args.checkpoint

    run_inference(config=cfg.config, split=args.split)
