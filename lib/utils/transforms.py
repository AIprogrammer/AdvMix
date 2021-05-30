# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch


def flip_back(output_flipped, matched_parts, args=None, cfg=None, dim=None):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4 or output_flipped.ndim == 3,\
        'output_flipped should be [batch_size, num_joints, height, width] or [batch_size, num_joints, 2]'
    
    if output_flipped.ndim == 4:
        output_flipped = output_flipped[..., ::-1]
    
    # flip x,y
    else:
        output_flipped = inv_coord_norm(output_flipped, cfg, args)
        if args.reg_coord:
            output_flipped[...,0] = cfg.MODEL.IMAGE_SIZE[0] - 1 - output_flipped[...,0] 
        else:
            output_flipped[...,0] = dim[3] - 1 - output_flipped[...,0]
        
        output_flipped = coord_norm(output_flipped, cfg, args).clone().cpu().numpy()

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], ...].copy()
        output_flipped[:, pair[0], ...] = output_flipped[:, pair[1], ...]
        output_flipped[:, pair[1], ...] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts; point index
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0] 
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

# the same rule: [1,0] [3,0] ==> [3,-2]  square triangle
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

# rotation coordination
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


def tofloat(x):

    if isinstance(x, np.ndarray):
        return torch.Tensor(x).cuda()
    
    if x.dtype == torch.float64 or x.dtype == torch.double:
        x = x.float()
    return x

def coord_norm(gt, cfg, args):
    
    gt = tofloat(gt)
    
    if args.reg_coord:
        image_size = [cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]]
    else:
        image_size = [cfg.MODEL.HEATMAP_SIZE[0], cfg.MODEL.HEATMAP_SIZE[1]]
    
    gt = (gt * 2 + 1) / torch.Tensor(image_size).cuda() - 1
    return gt

def inv_coord_norm(gt_norm, cfg, args):

    gt_norm = tofloat(gt_norm)
    
    if args.reg_coord:
        image_size = [cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]]
    else:
        image_size = [cfg.MODEL.HEATMAP_SIZE[0], cfg.MODEL.HEATMAP_SIZE[1]]
    
    gt = ((gt_norm + 1) * torch.Tensor(image_size).cuda() - 1) / 2
    return gt

def _tocuda(t):
    if isinstance(t, list):
        for index in range(len(t)):
            t[index] = _tocuda(t[index])
    else:
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t.copy()).cuda()
        elif t.is_cuda:
            return t
        else:
            t = torch.Tensor(t.clone()).cuda()
    return t

def _tocopy(t):
    if isinstance(t, list):
        for index in range(len(list)):
            t[index] = _tocopy(t[index])
    else:
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t.copy()).cuda()
        else:
            t = torch.Tensor(t.clone()).cuda()
    return t