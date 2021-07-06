from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import pprint
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--save_suffix',
                        help='model output dir suffix',
                        type=str,
                        default='')

    ### test robustness
    parser.add_argument('--test_robust',
                        help='normal test or test robustness',
                        default=False,
                        action='store_true')

    parser.add_argument('--corruption_type',
                    help='type of corruption',
                    type=str,
                    default='')

    parser.add_argument('--severity',
                    help='severity of corruption',
                    type=int,
                    default=0)

    # INPUT TYPE
    parser.add_argument('--dataset_root',
                        help='data directory, if only dataset root is provided, then all the images are processed',
                        type=str,
                        default='',
                        )
    parser.add_argument('--load_json_file',
                        help='load json file. The dataset root should also be given.',
                        type=str,
                        default='')
    # OUTPUT ROOT:
    parser.add_argument('--out_root',
                        help='data directory',
                        type=str,
                        default='/mnt/lustre/share/jinsheng/res_crop')

    parser.add_argument('--out_file',
                        help='data directory',
                        type=str,
                        default='res')

    # test & train
    parser.add_argument('--exp_id',
                        type=str,
                        default='')
    parser.add_argument('--load_from_G',
                        type=str,
                        default='')
    parser.add_argument('--load_from_D',
                        type=str,
                        default='')


    parser.add_argument('--sample_times',
                        type=int,
                        default=1)

    parser.add_argument('--adv_loss_weight',
                        type=float,
                        default=1)
    parser.add_argument('--combine_prob',
                        type=float,
                        default=0.2)
    parser.add_argument('--perturb_joint',
                        type=float,
                        default=0.2)
    parser.add_argument('--perturb_range',
                        type=int,
                        default=5)
    parser.add_argument('--sp_style',
                        type=float,
                        default=0)
    parser.add_argument('--advmix',
                    default=False,
                    action='store_true')

    parser.add_argument('--stylize_image',
                    default=False,
                    action='store_true')

    parser.add_argument('--joints_num',
                        type=int,
                        default=17)
    
    # generator
    parser.add_argument('--gen_input_chn',
                        type=int,
                        default=9)

    parser.add_argument('--downsamples',
                        type=int,
                        default=6)
    
    # knowledge distillation
    parser.add_argument('--kd_mseloss',
                    default=False,
                    action='store_true')

    parser.add_argument('--kd_klloss',
                    default=False,
                    action='store_true')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.1)

    # random corruption
    parser.add_argument('--random_corruption',
                    default=False,
                    action='store_true')
    
    args = parser.parse_args()

    return args