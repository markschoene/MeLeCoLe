import argparse
import h5py
import numpy as np

import neuroglancer
import neuroglancer.cli

"""

USAGE: python -i view.py path/to/object

"""


def get_args():
    parser = argparse.ArgumentParser(description="Computing predictions")
    parser.add_argument('--segmentation', type=str, default=None,
                        help='path to the computed segmenation file')
    parser.add_argument('--gt', type=str, required=True,
                        help='path to the grount truth segmenation file')
    parser.add_argument('--padding', type=int, default=(0, 0, 0), nargs='*',
                        help='padding of image around gt annotations')
    args = parser.parse_args()
    return args


def get_files(gt, result=None, padding=(0, 0, 0)):
    if result:
        with h5py.File(result, "r") as f:
            result = f['vol0'][:]
    with h5py.File(gt, "r") as f:
        if padding == (0, 0, 0):
            vol = f['volume'][:]
        else:
            vol = f['volume'][padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2]]
        gt = f['labels'][:]
        print(vol.shape, gt.shape)

    return result, gt.astype(np.uint64), vol


def add_layer(state, data, name):
    dimensions = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'],
                                              units='nm',
                                              scales=[30, 12, 12])

    state.dimensions = dimensions
    state.layers.append(
        name=name,
        layer=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=['x', 'y', 'z'],
                units=['nm', 'nm', 'nm'],
                scales=[30, 12, 12],
            ),
        )
    )


if __name__ == '__main__':
    viewer = neuroglancer.Viewer()
    args = get_args()
    print(f"Padding: {args.padding}")
    seg, gt, vol = get_files(result=args.segmentation, gt=args.gt, padding=args.padding)
    from skimage.segmentation import expand_labels
    seg = expand_labels(seg, distance=1)
    with viewer.txn() as s:
        add_layer(s, vol, 'volume')
        add_layer(s, gt, 'ground truth')
        if args.segmentation:
            add_layer(s, seg, 'segmentation')
    print(viewer)
