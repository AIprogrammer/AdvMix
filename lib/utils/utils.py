# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn


def create_logger(args, cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    cfg_name = args.save_suffix if args.save_suffix is not '' else cfg_name

    ### log output dir
    if args.test_robust:
        root_output_dir = Path('output_robustness')
        final_output_dir = root_output_dir / dataset / model / cfg_name / 'test_corruption'
    else:
        final_output_dir = root_output_dir / dataset / model / cfg_name
    
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')

    if args.test_robust:
        log_file = '{}_{}.log'.format(cfg_name, phase)
    else:
        log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)
    
    if args.test_robust:
        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / 'test_robustness' / \
            (cfg_name + '_' + time_str) / args.corruption_type / str(args.severity)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth', suffix=""):
    if suffix != "":
        torch.save(states, os.path.join(output_dir, filename[:-4] + '_' + suffix + ".pth"))
        if is_best and 'state_dict' in states:
            torch.save(states['best_state_dict'],
                    os.path.join(output_dir, 'model_best_{}.pth'.format(suffix)))
    else:
        torch.save(states, os.path.join(output_dir, filename))
        if is_best and 'state_dict' in states:
            torch.save(states['best_state_dict'],
                    os.path.join(output_dir, 'model_best.pth'))

def get_model_summary(model, *input_tensors, item_length=26, verbose=False, return_all=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds", "memory_access_cost"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)
            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0
            mac = 0
            mac_input = 0
            mac_output = 0
            mac_module = 0
            if class_name.find("Conv2d") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)
                    mac_module += param_.view(-1).size(0)

            flops = "Not Available"
            # 这种计算方式的定义是将一次乘法+加法 定义为一个FLOPs
            if class_name.find("Conv2d") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[-1]

            mac_input = input[0].view(-1).size(0)# input shape
            mac_output = output.view(-1).size(0)
            mac += (mac_input + mac_module + mac_output)

            # 与 batch 无关
            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops,
                    memory_access_cost=mac)
            )
        # 但凡是属于最小的module，那么就计算hook函数里面的内容，并返回mac，flops，参数量
        # 因此使用multi loss 时候， 实际计算值与计算图有关，而不是与init函数里面定义的module有关
        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}MAC(memory access cost){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)")), \
                ' ' * (space_len - len("MAC(memory access cost)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    mac_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        mac_sum += layer.memory_access_cost
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Memory Access Cost : {:,}".format(mac_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep

    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    if return_all:
        return [details, flops_sum, mac_sum, params_sum]
    return details
