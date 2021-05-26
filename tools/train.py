# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import sys
import _init_paths
from _init_parse import parse_args

from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train, train_advmix
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from torch.utils.data.dataset import ConcatDataset 


import dataset
import models


def main():
    args = parse_args()
    update_config(cfg, args)
    
    ### differ from the official args
    logger, final_output_dir, tb_log_dir = create_logger(
        args, cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    if args.advmix:
        model_teacher = copy.deepcopy(model)
        print('=> Traing adversarially.')
        model_G = models.Unet_generator.UnetGenerator(args.gen_input_chn,3,args.downsamples)
        print("=> UNet generator : {} input chanenels; {} downsample times".format(args.gen_input_chn, args.downsamples))
        model_G = torch.nn.DataParallel(model_G, device_ids=cfg.GPUS).cuda()

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    
    shutil.copy2(
        args.cfg,
        final_output_dir)

    shutil.copy2(
        'tools/train.py',
        final_output_dir)

    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    
    ## bug in tensorboardX
    try:
        writer_dict['writer'].add_graph(model, (dump_input, ))
    except:
        pass 
    try:
        logger.info(get_model_summary(model, dump_input))
    except:
        pass

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    if args.advmix:
        model_teacher = torch.nn.DataParallel(model_teacher, device_ids=cfg.GPUS).cuda()

    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, args, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    if cfg.DATASET.MINI_COCO:
        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, args, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, args, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    

    if args.advmix:
        if args.stylize_image:
            if cfg.DATASET.DATASET == 'mpii':
                style_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
                    cfg, args, 'data/stylize_image/output_mpii', cfg.DATASET.TRAIN_SET, True,
                    transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            else:
                style_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
                    cfg, args, 'data/stylize_image/output', cfg.DATASET.TRAIN_SET, True,
                    transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            train_dataset = ConcatDataset([train_dataset, style_dataset])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    last_epoch_G = -1
    optimizer = get_optimizer(cfg, model)
    if args.advmix:
        optimizer_G = get_optimizer(cfg, model_G)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint_D.pth'
    )

    checkpoint_file_G = os.path.join(
        final_output_dir, 'checkpoint_G.pth'
    )


    if os.path.exists(args.load_from_D):
        logger.info('=> Fine tuning by loading pretrained model: {}'.format(args.load_from_D))
        pretrained_dict = torch.load(args.load_from_D)
        pretrained_dict = {'module.' + k:v for k,v in pretrained_dict.items()}
        share_state = {}
        model_state = model.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                share_state[k] = v
        
        logger.info('Model dict :{}; shared dict :{}'.format(len(model_state), len(share_state)))
        model_state.update(share_state)
        model.load_state_dict(model_state)

    if os.path.exists(args.load_from_D) and args.advmix:
        logger.info('=> Build teacher model by loading pretrained model: {}'.format(args.load_from_D))
        pretrained_dict = torch.load(args.load_from_D)
        pretrained_dict = {'module.' + k:v for k,v in pretrained_dict.items()}
        share_state = {}
        model_state = model_teacher.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                share_state[k] = v
        logger.info('Model dict :{}; shared dict :{}'.format(len(model_state), len(share_state)))
        model_state.update(share_state)
        model_teacher.load_state_dict(model_state)

    if os.path.exists(args.load_from_G):
        pretrained_dict = torch.load(args.load_from_G)
        pretrained_dict = {'module.' + k:v for k,v in pretrained_dict.items()}
        share_state = {}
        model_state = model_G.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                share_state[k] = v
        
        model_state.update(share_state)
        model_G.load_state_dict(model_state)
    

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file) and args.advmix:
        logger.info("=> loading checkpoint teacher model'{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model_teacher.load_state_dict(checkpoint['state_dict'])


    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file_G) and args.advmix:
        logger.info("=> loading checkpoint generator'{}'".format(checkpoint_file_G))
        checkpoint_G = torch.load(checkpoint_file_G)
        begin_epoch_G = checkpoint_G['epoch']
        best_perf_G = checkpoint_G['perf']
        last_epoch_G = checkpoint_G['epoch']
        model_G.load_state_dict(checkpoint_G['state_dict'])

        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file_G, checkpoint_G['epoch']))


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    if args.advmix:
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch_G
        )

    # name_values, perf_indicator = validate(
    #     cfg, args, valid_loader, valid_dataset, model, criterion,
    #     final_output_dir, tb_log_dir, writer_dict
    # )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()
        if args.advmix:
            lr_scheduler_G.step()
        
        print('=> The learning rate is: ', optimizer.param_groups[0]['lr'], optimizer_G.param_groups[0]['lr'])
        
        if args.advmix:
            train_advmix(cfg, args, train_loader, [model, model_G, model_teacher], criterion, [optimizer, optimizer_G], epoch,
                final_output_dir, tb_log_dir, writer_dict)
        else:
            print('=> Normal training ...')
            train(cfg, args, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict)
        
        name_values, perf_indicator = validate(
            cfg, args, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False
        
        logger.info("==> best mAP is {}".format(best_perf))
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, suffix="D")

        if args.advmix:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model_G.state_dict(),
                'best_state_dict': model_G.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer_G.state_dict(),
            }, best_model, final_output_dir, suffix = "G")

        
    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
