# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os, copy

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds, get_final_preds_using_softargmax, SoftArgmax2D
from utils.transforms import flip_back, tofloat, coord_norm, inv_coord_norm, _tocopy, _tocuda
from utils.vis import save_debug_images
import dsntnn
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def train(config, args, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if isinstance(model, list):
        model = model[0].train()
        model_D = model[1].train()
    else:
        model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in tqdm(enumerate(train_loader)):

        data_time.update(time.time() - end)

        outputs = model(input)

        target = target[0].cuda(non_blocking=True)
        target_hm = target
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(outputs, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        _, avg_acc, cnt, pred = accuracy(outputs,
                                        target, args=None, cfg=config)
                                        
        outputs = _tocuda(outputs)

        acc.update(avg_acc, cnt)


        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

            save_debug_images(config, input, meta, target_hm, pred*4, outputs,
                            prefix)


def set_require_grad(nets, requires_grad=True):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def train_advmix(config, args, train_loader, models, criterion, optimizers, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if isinstance(models, list):
        model = models[0].train()
        model_G = models[1].train()
        model_teacher = models[2].eval()
    else:
        models.train()

    optimizer = optimizers[0]
    optimizer_G = optimizers[1]

    end = time.time()
    for i, (inputs, targets, target_weights, metas) in tqdm(enumerate(train_loader)):

        data_time.update(time.time() - end)
        # mask_channel = meta['model_supervise_channel'] > 0.5
        if isinstance(inputs, list):
            inputs = [_.cuda(non_blocking=True) for _ in inputs]
            target = targets[0].cuda(non_blocking=True)
            target_weight = target_weights[0].cuda(non_blocking=True)
            meta = metas[0]
        else:
            inputs = inputs.cuda(non_blocking=True)
        
        G_input = torch.cat(inputs, dim=1)
        mix_weight = F.softmax(model_G(G_input), dim=1)

        set_require_grad(model, True)
        optimizer.zero_grad()
        tmp = inputs[0] * mix_weight[:,0,...].unsqueeze(dim=1)
        for list_index in range(1, len(inputs)):
            tmp += inputs[list_index] * mix_weight[:,list_index].unsqueeze(dim=1)

        D_output_detach = model(tmp.detach())

        with torch.no_grad():
            teacher_output = model_teacher(inputs[0])

        loss_D_hm = criterion(D_output_detach, target, target_weight)
        loss_D_kd = criterion(D_output_detach, teacher_output, target_weight)
        loss_D = loss_D_hm * (1 - args.alpha) + loss_D_kd * args.alpha
        loss_D.backward()
        optimizer.step()

        # G: compute gradient and do update step
        set_require_grad(model, False)
        optimizer_G.zero_grad()
        outputs = model(tmp)
        output = outputs
        loss_G = -criterion(output, target, target_weight) * args.adv_loss_weight
        loss_G.backward()
        optimizer_G.step()
    
        # measure accuracy and record loss
        losses.update(loss_D.item(), inputs[0].size(0))
        _, avg_acc, cnt, pred = accuracy(output,
                                        target, args=None, cfg=config)

        acc.update(avg_acc, cnt)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inputs[0].size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, inputs[0], copy.deepcopy(meta), target, pred*4, outputs,
                            prefix + '_clean')
            save_debug_images(config, tmp, copy.deepcopy(meta), target, pred*4, outputs,
                            prefix)


def validate(config, args, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, cpu=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    feat_dict = {}

    with torch.no_grad():
        end = time.time()
        time_gpu = 0.
        for i, (input, target, target_weight, meta) in tqdm(enumerate(val_loader)):
            if not cpu:
                input = input.cuda()
            # compute output
            torch.cuda.synchronize()
            infer_start = time.time()
            outputs = model(input)

            infer_end = time.time()
            torch.cuda.synchronize()
            time_gpu += (infer_end - infer_start)

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped
                
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                        val_dataset.flip_pairs)
                if not cpu:
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                else:
                    output_flipped = torch.from_numpy(output_flipped.copy())
                    
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                output = (output + output_flipped) * 0.5
                
            if not cpu:
                target = target[0].cuda(non_blocking=True)
                target_hm = target
                target_weight = target_weight.cuda(non_blocking=True)
        
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            
            _, avg_acc, cnt, pred = accuracy(output,
                                             target, args=None, cfg=config)

            output = _tocuda(output)
            acc.update(avg_acc, cnt)
            batch_time.update(time.time() - end)
            end = time.time()

            # corresponding center scale joint
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            

            preds, maxvals = get_final_preds(
                config, args, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )

                save_debug_images(config, input, meta, target_hm, pred * 4, output,
                                prefix) 
        
        print('=> The average inference time is :', time_gpu / len(val_loader))
    
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1
    
    return name_values, perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
