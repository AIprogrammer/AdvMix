# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

import collections
from _init_parse import parse_args
import pandas as pd

def val_model_init():

    # adjust the gpu_ids
    args = parse_args()
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    return model

def val(distortion_name, severity, model):
    args = parse_args()
    args.corruption_type = distortion_name
    args.severity = severity

    update_config(cfg, args)

    exp_id = args.exp_id
    which_dataset = cfg.DATASET.DATASET

    logger, final_output_dir, tb_log_dir = create_logger(args, 
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))

    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, args, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    name_values, perf_indicator = validate(cfg, args, valid_loader, valid_dataset, model, criterion,
                        final_output_dir, tb_log_dir)
    

    # recording overall results
    overall_dir = os.path.join(final_output_dir, 'robust_C.val')
    record = open(os.path.join(final_output_dir, 'robust_C.val'), 'a')
    record.write(distortion_name + '_' + str(severity) + ':' + '\t')
    for keys, values in name_values.items():
        record.write(keys + ' = ' + str(values) + '\t')
    record.write('\n')
    record.close()
    return name_values, perf_indicator, final_output_dir, exp_id, which_dataset, overall_dir

def get_corrpution_results():
    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    res = []
    model = val_model_init()
    name_values, perf_indicator, final_output_dir, exp_id, which_dataset, overall_dir = val('clean', 0, model)
    res.append(perf_indicator)

    for distortion_name in distortions[:15]:
        for severity in range(5):
            name_values, perf_indicator, final_output_dir, exp_id, which_dataset, overall_dir = val(distortion_name, severity, model)
            res.append(perf_indicator)
    
    if which_dataset == 'mpii':
        get_final_results_mpii(res, distortions, final_output_dir, exp_id, mode='td')
    else:
        mode = 'bu' if cfg.model.type == 'BottomUp' else 'td'
        get_final_results(res, distortions, final_output_dir, exp_id, mode=mode)

def get_final_results(mAP, distortions, final_output_dir, exp_id,mode='td'):
    dic = {}
    assert len(mAP) == 96, 'Result length'
    dic['clean_mAP'] = [mAP.pop(0)]
    print(mAP)

    all_tmp = 0
    for dis in distortions:
        tmp = []
        for i in range(5):
            tmp.append(mAP[distortions.index(dis) * 5 + i])
        dic[dis] = [sum(tmp) / len(tmp)]
        if dis in distortions[:15]:
            all_tmp += dic[dis][0]
    
    dic['mean_corrupted_AP'] = [all_tmp / 15]
    dic['rAP'] = dic['mean_corrupted_AP'][0] / dic['clean_mAP'][0]

    dataframe = pd.DataFrame(dic)
    columns = ['clean_mAP', 'mean_corrupted_AP', 'rAP'] + distortions 
    dataframe.to_csv(final_output_dir + '/' + exp_id + ".csv", index=False,sep=',', columns=columns)

def get_final_results_mpii(mean, distortions, final_output_dir, exp_id,mode='td'):
    dic = {}
    assert len(mean) == 96, 'Result length'
    dic['clean_mean'] = [round(mean.pop(0),3)]
    print(mean)

    all_tmp = 0
    for dis in distortions:
        tmp = []
        for i in range(5):
            tmp.append(mean[distortions.index(dis) * 5 + i])
        dic[dis] = [round(sum(tmp) / len(tmp),3)]
        if dis in distortions[:15]:
            all_tmp += dic[dis][0]
    
    dic['mean_corrupted_mean'] = [round(all_tmp / 15,3)]
    dic['rmean'] = round(dic['mean_corrupted_mean'][0] / dic['clean_mean'][0],3)

    dataframe = pd.DataFrame(dic)
    columns = ['clean_mean', 'mean_corrupted_mean', 'rmean'] + distortions 
    dataframe.to_csv(final_output_dir + '/' + exp_id + ".csv", index=False,sep=',', columns=columns)

if __name__ == '__main__':
    get_corrpution_results()
