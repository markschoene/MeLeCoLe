from partition_comparison import variation_of_information as voi
import h5py
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from cc3d import connected_components
from skimage.measure import label
from skimage.segmentation import expand_labels


def get_args():
    parser = argparse.ArgumentParser(description="Computing predictions")
    parser.add_argument('--segmentation', type=str, required=True,
                        help='path to the computed segmenation file')
    parser.add_argument('--gt', type=str, required=True,
                        help='path to the grount truth segmenation file')
    parser.add_argument('--membranes', type=str, required=False, default=None)
    args = parser.parse_args()
    return args


def get_files(result, gt, membranes=None):
    with h5py.File(result, "r") as f:
        r = f['vol0'][:]
    with h5py.File(gt, "r") as f:
        g = f['labels'][:].astype(np.uint64)
    if membranes:
        with h5py.File(membranes, 'r') as f:
            m = f['vol0'][:]
    else:
        m = None

    return r, g, m


def print_cell_size(segmentation, gt, save_dir, n_bins=11, plot_alpha=0.6):
    _, seg_counts = np.unique(segmentation, return_counts=True)
    _, gt_counts = np.unique(gt, return_counts=True)

    x_max = np.max([segmentation.max(), gt.max()])
    bins = np.exp(np.arange(n_bins) * np.log(x_max) / (n_bins - 1))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist(gt_counts, bins, alpha=1, label='GT')
    ax.hist(seg_counts, bins, alpha=plot_alpha, label='Seg')
    ax.set_xscale('log')
    ax.legend()
    fig.savefig(os.path.join(save_dir, 'cell_histogram.png'))


if __name__ == '__main__':
    args = get_args()
    result, gt, mem = get_files(args.segmentation, args.gt, args.membranes)

    # recompute connected components

    mask = gt == 0
    cc = connected_components(gt, connectivity=6)
    cc[mask] = 0
    gt = cc.astype(np.uint64)

    print("Resulting VOI: ", voi(result.flatten(), gt.flatten()))
    print(f"GT cells: {len(np.unique(gt))}, Segmented cells: {len(np.unique(result))}")
    if args.membranes:
        mem = (mem > 0.99).astype(np.uint64)
        connected_components_segmentation = connected_components(mem, connectivity=6).astype(np.uint64)
        print("Connected components VOI: ", voi(connected_components_segmentation.flatten(), gt.flatten()))
    print_cell_size(segmentation=result,
                    gt=gt,
                    save_dir=os.path.dirname(args.segmentation),
                    n_bins=21,
                    plot_alpha=0.5)
