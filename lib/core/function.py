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


def init_w():
    import numpy as np
    w_pred = np.zeros()
    w_target_norm = np.zeros()
    w_target_inv = np.zeros()


def joint_hm_coord(args, cfg, model, criterion, outputs, target, target_weight, istrain=True):

    if isinstance(outputs, list) and isinstance(target, list) and args.dsntnn:
        # outputs[0] is in [-1,1]
        coord_loss = dsntnn.euclidean_losses(outputs[0], coord_norm(target[0], cfg, args))
        r"savebug"
        np.save('debug/pred.npy', outputs[0].detach().cpu().numpy())
        np.save('debug/target_norm.npy', coord_norm(target[0], cfg, args).detach().cpu().numpy())
        np.save('debug/target_inv.npy', inv_coord_norm(coord_norm(target[0], cfg, args), cfg, args).detach().cpu().numpy())
        np.save('debug/target.npy', target_weight[0].detach().cpu().numpy())
        # gt: y, x
        only_use_mse = False
        if only_use_mse:
            hm_loss = criterion(outputs[1], target[-1], target_weight)
            lambda_coord, lambda_hm = args.lambda_w
            loss = dsntnn.average_loss(coord_loss * lambda_coord + hm_loss * lambda_hm, mask=target_weight.squeeze())
        else:
            hm_loss = dsntnn.js_reg_losses(outputs[-1], coord_norm(target[0], cfg, args), sigma_t=1.0)
            lambda_coord, lambda_hm = args.lambda_w
            loss = dsntnn.average_loss(coord_loss * lambda_coord + hm_loss * lambda_hm, mask=target_weight.squeeze())
            conf = torch.sigmoid(1.0 / dsntnn.variance_reg_losses(outputs[-1], sigma_t=1.0))

        output = outputs[-1]
    
    elif isinstance(outputs, list) and args.softargmax:
        # get max num by softargmax
        loss = criterion(outputs[0], coord_norm(target[0], cfg, args), target_weight)
    
    elif isinstance(outputs, list) and args.fc_coord: # only coord loss
        loss = criterion(outputs[0], coord_norm(target[0], cfg, args), target_weight)
    else:
        output = outputs
        loss = criterion(output, target, target_weight)
    
    if args.return_conf:
        return [loss, conf]
    else:
        return loss

def updateBN(model, weight_decay):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(weight_decay * torch.sign(m.weight.data))  # L1 

def train(config, args, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, sparse_bn=False, weight_decay=0):
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
        if args.dsntnn or args.fc_coord or args.softargmax:
            assert len(target) == 2, 'generate coord and heatmap'  # [hm, coord]
            target_hm = target[0].cuda(non_blocking=True) # heatmap

            # attention: meta['joints'] or the target['heatmap_joint']
            if args.reg_coord:
                target = [meta['joints'].cuda(non_blocking=True)[...,:2], target_hm]
            else:
                target = [target[1].cuda(non_blocking=True)[..., :2], target_hm]
        
        else:
            target = target[0].cuda(non_blocking=True)
            target_hm = target
        
        target_weight = target_weight.cuda(non_blocking=True)
        loss = joint_hm_coord(args, config, model, criterion, outputs, target, target_weight)

        if args.return_conf:
            loss = loss[0]
        
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        # L1 regularization on BN layer
        if sparse_bn:
            updateBN(model, weight_decay)
        
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        
        if not args.dsntnn and not args.fc_coord:
            args_tmp = None
        else:
            args_tmp = args
        _, avg_acc, cnt, pred = accuracy(outputs,
                                        target, args=args_tmp, cfg=config)
                                        
        # print('==> debug', outpuoptimizerts.shape, pred.shape)
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

            # 3_27 update
            if args.dsntnn or args.fc_coord or args.softargmax:
                preds_tmp = inv_coord_norm(outputs[0], config, args).clone().detach().cpu().numpy()

                '''
                def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix, coord=None):
                '''
                if args.reg_coord:
                    save_debug_images(config, input, meta, target_hm, preds_tmp, outputs[1:],
                                    prefix)
                else:
                    # cooresponding the heatmap points
                    save_debug_images(config, input, meta, target_hm, pred * 4, outputs[1:],
                                    prefix, coord=preds_tmp)

            else:
                save_debug_images(config, input, meta, target_hm, pred*4, outputs,
                                prefix)


def set_require_grad(nets, requires_grad=True):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def train_adv(config, args, train_loader, models, criterion, optimizers, epoch,
          output_dir, tb_log_dir, writer_dict, sparse_bn=False, weight_decay=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if isinstance(models, list):
        model = models[0].train()
        model_G = models[1].train()
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
        
        mix_weight = F.softmax(model_G(inputs[0]), dim=1) # conv mix weight; only using clean image

        set_require_grad(model, True)
        optimizer.zero_grad()
        tmp = inputs[0] * mix_weight[:,0,...].unsqueeze(dim=1)
        for list_index in range(1, len(inputs)):
            tmp += inputs[list_index] * mix_weight[:,list_index].unsqueeze(dim=1)

        D_output_detach = model(tmp.detach()) # mixed img, return heatmap

        loss_D = criterion(D_output_detach, target, target_weight)
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

        if not args.dsntnn and not args.fc_coord:
            args_tmp = None
        else:
            args_tmp = args
        _, avg_acc, cnt, pred = accuracy(output,
                                        target, args=args_tmp, cfg=config)

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

            if args.dsntnn or args.fc_coord or args.softargmax:
                preds_tmp = inv_coord_norm(outputs[0], config, args).clone().detach().cpu().numpy()
                '''
                def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix, coord=None):
                '''
                if args.reg_coord:
                    save_debug_images(config, inputs[0], meta, target, preds_tmp, outputs[1:],
                                    prefix)
                else:
                    save_debug_images(config, inputs[0], meta, target, pred * 4, outputs[1:],
                                    prefix, coord=preds_tmp)
            else:
                save_debug_images(config, inputs[0], copy.deepcopy(meta), target, pred*4, outputs,
                                prefix + '_clean')
                save_debug_images(config, tmp, copy.deepcopy(meta), target, pred*4, outputs,
                                prefix)


def train_adv_kd(config, args, train_loader, models, criterion, optimizers, epoch,
          output_dir, tb_log_dir, writer_dict, sparse_bn=False, weight_decay=0):
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

        if not args.dsntnn and not args.fc_coord:
            args_tmp = None
        else:
            args_tmp = args
        _, avg_acc, cnt, pred = accuracy(output,
                                        target, args=args_tmp, cfg=config)

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

            if args.dsntnn or args.fc_coord or args.softargmax:
                preds_tmp = inv_coord_norm(outputs[0], config, args).clone().detach().cpu().numpy()
                '''
                def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix, coord=None):
                '''
                if args.reg_coord:
                    save_debug_images(config, inputs[0], meta, target, preds_tmp, outputs[1:],
                                    prefix)
                else:
                    save_debug_images(config, inputs[0], meta, target, pred * 4, outputs[1:],
                                    prefix, coord=preds_tmp)
            else:
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

            if 0:
                print(input.shape, model.module.extract_feat['conv2'].shape)
                for k_i in range(input.shape[0]):
                    if meta['instance_index'][k_i] not in feat_dict:
                        feat_dict[meta['instance_index'][k_i]] = {}
                    
                    feat_dict[meta['instance_index'][k_i]]['conv2'] = model.module.extract_feat['conv2'][k_i].detach().cpu().numpy()
                    feat_dict[meta['instance_index'][k_i]]['layer1'] = model.module.extract_feat['layer1'][k_i].detach().cpu().numpy()
                    feat_dict[meta['instance_index'][k_i]]['stage2'] = model.module.extract_feat['stage2'][k_i].detach().cpu().numpy()
                    feat_dict[meta['instance_index'][k_i]]['stage3'] = model.module.extract_feat['stage3'][k_i].detach().cpu().numpy()
                    feat_dict[meta['instance_index'][k_i]]['stage4'] = model.module.extract_feat['stage4'][k_i].detach().cpu().numpy()
                    feat_dict[meta['instance_index'][k_i]]['hm'] = model.module.extract_feat['hm'][k_i].detach().cpu().numpy()


            infer_end = time.time()
            torch.cuda.synchronize()
            time_gpu += (infer_end - infer_start)

            if not args.dsntnn and not args.fc_coord:
                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs
            else:
                output = outputs
            
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if not args.dsntnn and not args.fc_coord or args.softargmax:

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
                
                else:
                    dim = outputs_flipped[1].shape
                    for index, item_o in enumerate(outputs_flipped):
                        outputs_flipped[index] = flip_back(item_o.cpu().numpy(),
                                                val_dataset.flip_pairs, args=args, cfg=config, dim=dim)
                    
                    if not cpu:
                        outputs_flipped = _tocuda(outputs_flipped)
                    else:
                        outputs_flipped = _tocopy(outputs_flipped)
                    
                    if config.TEST.SHIFT_HEATMAP:
                        outputs_flipped[1][..., 1:] = \
                            outputs_flipped[1].clone()[..., 0:-1]
                    
                    for index in range(len(output)):
                        output[index] = (output[index] + outputs_flipped[index]) * 0.5
                
            if not cpu:
                if args.dsntnn or args.fc_coord:
                    # [hm, coord] ==> [coord, hm]
                    assert len(target) == 2, 'generate coord and heatmap'  # [hm, coord]
                    target_hm = target[0].cuda(non_blocking=True)

                    if args.reg_coord:
                        target = [meta['joints'].cuda(non_blocking=True)[...,:2], target_hm]
                    else:
                        target = [target[1].cuda(non_blocking=True)[..., :2], target_hm]
                else:
                    target = target[0].cuda(non_blocking=True)
                    target_hm = target
     
                target_weight = target_weight.cuda(non_blocking=True)
        
            r"""output = outputs
            """
            loss = joint_hm_coord(args, config, model, criterion, output, target, target_weight, istrain=False)
            
            if args.return_conf:
                conf = loss[1]
                loss = loss[0]
            
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            
            if not args.dsntnn and not args.fc_coord:
                args_tmp = None
            else:
                args_tmp = args
            
            _, avg_acc, cnt, pred = accuracy(output,
                                             target, args=args_tmp, cfg=config)

            output = _tocuda(output)
            acc.update(avg_acc, cnt)
            batch_time.update(time.time() - end)
            end = time.time()

            # corresponding center scale joint
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            
            r"""
            loss | accuracy | mAP | real_output
            """
            # for training ==> attention the vis pred * 4 or not
            # for validation ==> vis pred * 4 or not | final coord ==> reg_coord or not
            if args.dsntnn or args.fc_coord or args.softargmax:
                # only use regularization loss; not use coord loss
                if args.lambda_w[0] == 0:
                    print('==> the weight of coord loss is zero')
                    output[-1] = inv_coord_norm(output[-1], config, args).clone().cpu().numpy()
                    preds, maxvals = get_final_preds(
                        config, args, output[-1].clone().cpu().numpy(), c, s)
                else:
                    # regress the normalized coord
                    preds_tmp = inv_coord_norm(output[0], config, args).clone().cpu().numpy()
                    preds, maxvals = get_final_preds(
                        config, args, output[-1].clone().cpu().numpy(), c, s, cal_hm_coord=False, coord=preds_tmp, reg_hm=not args.reg_coord)
            else:
                preds, maxvals = get_final_preds(
                    config, args, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            
            if args.return_conf:
                maxvals = torch.unsqueeze(conf, dim=-1).cpu().numpy()
            try:
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
            except: # all_preds
                all_preds[idx:idx + num_images, :, 2:3] = maxvals[:,:all_preds.shape[1], :]
                
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

                if args.dsntnn or args.fc_coord or args.softargmax:
                    preds_tmp = inv_coord_norm(output[0], config, args).clone().detach().cpu().numpy()

                    if args.reg_coord:
                        save_debug_images(config, input, meta, target_hm, preds_tmp, outputs[1:],
                                        prefix)
                    else:
                        save_debug_images(config, input, meta, target_hm, pred * 4, outputs[1:],
                                        prefix, coord=preds_tmp)
                else:
                    save_debug_images(config, input, meta, target_hm, pred * 4, output,
                                    prefix) 
        
        print('the average inference time is :', time_gpu / len(val_loader))
    
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
    
    if 0:
        print(output_dir)
        print(len(feat_dict.keys()))
        save_dict = {}
        for kid, key in enumerate(feat_dict.keys()):
            if kid >= 500:
                break
            save_dict[key] = feat_dict[key]
            
        torch.save(save_dict, output_dir + '/' + 'feat_dict_{}_{}.pth'.format(args.corruption_type, args.severity))

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
