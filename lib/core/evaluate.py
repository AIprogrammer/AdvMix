# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds
from utils.transforms import flip_back, tofloat, coord_norm, inv_coord_norm, _tocopy, _tocuda

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(outputs, target, hm_type='gaussian', thr=0.5, args=None, cfg=None):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    if isinstance(outputs, list):
        for index in range(len(outputs)):
            outputs[index] = outputs[index].clone().detach().cpu().numpy()
        idx = list(range(outputs[-1].shape[1]))

    else:
        outputs = outputs.clone().detach().cpu().numpy()
        idx = list(range(outputs.shape[1]))
    
    if isinstance(target, list):
        for index in range(len(target)):
            target[index] = target[index].clone().detach().cpu().numpy()
        idx = list(range(target[-1].shape[1]))

    else:
        target = target.clone().detach().cpu().numpy()
        idx = list(range(target.shape[1]))
    
    norm = 1.0
    
    if hm_type == 'gaussian' and args is None:
        pred, _ = get_max_preds(outputs)
        target, _ = get_max_preds(target)
        h = outputs.shape[2] # y
        w = outputs.shape[3] # x

    else:
        assert outputs[0].ndim == 3, 'the output coord must be 3 dims'
        pred = outputs[0]
        pred = inv_coord_norm(pred, cfg, args).clone().detach().cpu().numpy()
        target, _ = get_max_preds(target[-1])
        h = outputs[-1].shape[2]
        w = outputs[-1].shape[3]

    norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


