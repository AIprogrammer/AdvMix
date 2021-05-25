# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, smooth_L1=False):
        super(JointsMSELoss, self).__init__()
        if smooth_L1:
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            self.criterion = nn.SmoothL1Loss()
        
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)

        # try:
        #     if isinstance(output, list):
        #         output = torch.Tensor(output).cuda(non_blocking=True)
            
        #     if isinstance(target, list):
        #         target = torch.Tensor(target).cuda(non_blocking=True)

        # except:
        #     print('='*20)
        #     print(type(output), type(target), output.shape, target.shape)
        #     print(output)
        #     print('='*20)
        #     print(target)

        def tofloat(x):
            if x.dtype == torch.float64 or x.dtype == torch.double:
                x = x.float()
            return x
        
        output = tofloat(output)
        target = tofloat(target)
        target_weight = tofloat(target_weight)

        if output.dim() == 4:
            heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)  # size of split, dim
            heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        else:
            heatmaps_pred = output # bct, 17, 2
            heatmaps_gt = target # bct, 17,2
        
        loss = 0

        # print(output.dtype, target.dtype, target_weight.dtype, heatmaps_pred.dtype, heatmaps_gt.dtype, )
        # []
        for idx in range(num_joints):
            if not isinstance(heatmaps_pred, tuple) and heatmaps_pred.shape[2] == 2:
                heatmap_pred = heatmaps_pred[:,idx].squeeze()
                heatmap_gt = heatmaps_gt[:,idx].squeeze()

            else:
                heatmap_pred = heatmaps_pred[idx].squeeze()
                heatmap_gt = heatmaps_gt[idx].squeeze()
            
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
