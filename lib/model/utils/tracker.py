import numpy as np
import torch
from model.rpn.bbox_transform import  bbox_overlaps


class Tracker:
    def __init__(self):
        # Bboxes info
        self.bboxes = []
        self.bboxes_ids = []

    def init_bboxes_ids(self, bboxes):
        for idx, bbox in enumerate(bboxes):
            self.bboxes.append(bbox)
            self.bboxes_ids.append(idx)


def matrix_binarize_to_max(mat):
    mat_bin_rows = np.zeros_like(mat)
    mat_bin_cols = np.zeros_like(mat)
    mat_bin = np.zeros_like(mat)

    max_val_idx_rows = np.argmax(mat, axis=0)
    max_val_idx_cols = np.argmax(mat, axis=1)

    for i in range(len(max_val_idx_rows)):
        mat_bin_rows[max_val_idx_rows[i], i] = 1

    for i in range(len(max_val_idx_cols)):
        mat_bin_cols[i, max_val_idx_cols[i]] = 1

    # Intersection
    m = mat_bin_rows + mat_bin_cols
    mat_val_idx = np.where(m == 2)
    mat_bin[mat_val_idx] = 1

    return mat_bin


if __name__ == "__main__":
    prev_bboxes = [[3.76792908e-01, 4.34368958e+02, 7.63530807e+01, 6.57716125e+02],
    [8.32442627e+02, 3.39747192e+02, 8.98434814e+02, 4.35743286e+02],
    [1.06633594e+03, 3.24260712e+02, 1.14194043e+03, 4.41318451e+02],
    [1.01811255e+03, 4.21633728e+02, 1.10284790e+03, 5.64207825e+02],
    [9.94417969e+02, 2.33804977e+02, 1.04186536e+03, 3.17149719e+02],
    [2.29070511e+01, 6.78276794e+02, 1.51329590e+02, 8.79110657e+02],
    [2.54017044e+02, 3.37344086e+02, 3.39777863e+02, 4.85414520e+02],
    [1.24849823e+02, 3.75690308e+02, 1.98401672e+02, 5.11423096e+02],
    [9.95774170e+02, 9.70689392e+01, 1.03487732e+03, 1.63836304e+02],
    [9.78262253e+01, 3.65794769e+02, 1.47269806e+02, 4.82248749e+02],
    [9.65077209e+02, 9.76259308e+01, 9.98922668e+02, 1.62632736e+02]]

    bboxes = [[6.7933655e-01, 4.4242392e+02, 6.9897736e+01, 6.5446362e+02],
    [1.3067049e+02, 3.6737073e+02, 1.9313533e+02, 5.0843341e+02],
    [2.2075668e+01, 6.7981305e+02, 1.1169687e+02, 8.8720721e+02],
    [9.2484413e+01, 3.6561328e+02, 1.4793463e+02, 4.8632440e+02],
    [9.9439569e+02, 2.3358698e+02, 1.0447622e+03, 3.2090771e+02],
    [9.9848260e+02, 1.0032078e+02, 1.0356556e+03, 1.6427917e+02],
    [9.6836499e+02, 1.0141615e+02, 9.9535608e+02, 1.6417345e+02],
    [1.0373208e+03, 3.1378656e+02, 1.1194871e+03, 4.4549713e+02],
    [9.9250964e+02, 4.2929956e+02, 1.0730536e+03, 5.7156744e+02]]

    tracker = Tracker()
    tracker.init_bboxes_ids(prev_bboxes)

    prev_bboxes_th = torch.tensor(prev_bboxes)
    bboxes_th = torch.tensor(bboxes)
    overlaps_th = bbox_overlaps(bboxes_th, prev_bboxes_th)

    overlaps = overlaps_th.numpy()
    overlaps_bin = matrix_binarize_to_max(overlaps)

    print(overlaps_bin)
