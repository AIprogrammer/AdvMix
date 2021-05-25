import numpy as np
import os
from os.path import join
import argparse
from tqdm import tqdm
# from thop import clever_format, profile
import json
import warnings
import re
from collections import defaultdict

# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def conv_dw_wo_bn(inp, oup, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x1 = x.view(batchsize, groups,
        channels_per_group, height, width)

    x2 = torch.transpose(x1, 1, 2).contiguous()

    # flatten
    x3 = x2.view(batchsize, -1, height, width)

    return x3


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, gp=False, shuffle=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.gp = gp
        self.shuffle = shuffle
        if gp:
            self.conv = conv_dw_wo_bn(inp_dim, out_dim, kernel_size, stride)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        # assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        # add channel shuffle
        group = 2
        if x.data.size()[1] % group == 0 and self.gp and self.shuffle:
            x = channel_shuffle(x, 2)
        return x

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class light_model(nn.Module):
    def __init__(self, settings):
        super(light_model, self).__init__()
        oup_dim = 17

        self.settings = settings
        self.innercat = self.settings['innercat']
        self.crosscat = self.settings['crosscat']
        self.upsample4x = self.settings['upsample4x']

        self.deconv_with_bias = False

        # head block
        self.pre = nn.Sequential(
            Conv(3, 12, 3, 2, bn=True),
            Conv(12, 24, 3, 2, bn=True),
            Conv(24, 24, 3, 1, bn=True, gp=settings['pre2'][0], shuffle=settings['pre2'][1]),
            Conv(24, 24, 3, 1, bn=True, gp=settings['pre3'][0], shuffle=settings['pre3'][1])
        )
        self.conv3_1_stage2 = Conv(24, 36, 3, 2, bn=True, gp=settings['conv_3_1'][0], shuffle=settings['conv_3_1'][1])
        self.conv4_stage2 = nn.Sequential(
            Conv(36, 24, 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1]),
            Conv(24, 24, 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1])
        )

        if self.upsample4x:
            self.stage2_pre_up4x = nn.Sequential(
                Conv(24, 64, 3, 2, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, relu=False, ),
                nn.Upsample(scale_factor=4, mode='bilinear')
            )
        else:
            # body block
            self.stage2_pre = nn.Sequential(
                Conv(24, 64, 3, 2, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, ),
                Conv(64, 64, 3, 1, bn=True, relu=False, ),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            if self.innercat:
                self.Cconv1_stage2 = Conv(24, 64, 1, 1, bn=True, relu=False)
                # concat self.stage2_pre & self.Cconv1_stage2

            if self.crosscat:
                self.Mconv2_stage2 = nn.Sequential(
                    Conv(64, 64, 3, 1, bn=False, relu=False, gp=settings['Mconv2_stage'][0],
                         shuffle=settings['Mconv2_stage'][1]),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )

        self.Crossconv1_stage2 = Conv(24, 64, 1, 1, bn=True, relu=False)
        # concat self.Mconv2_stage2 & self.Crossconv1_stage2

        self.stage2_post = nn.Sequential(  
            Conv(64, 36, 1, 1, bn=True),
            Conv(36, 36, 3, 1, bn=True, gp=settings['stage2_post'][0], shuffle=settings['stage2_post'][1]),
            Conv(36, 64, 1, 1, bn=True)
        )

        self.final_layer = Conv(64, oup_dim, 1, 1, bn=False, relu=False)  # or bilinear downsample

        self.init_weights()

    def forward(self, x: Variable):

        conv3 = self.pre(x)
        conv3_1 = self.conv3_1_stage2(conv3)

        cat_s2 = self.conv4_stage2(conv3_1)
        
        if self.upsample4x:

            innercat_s2 = self.stage2_pre_up4x(cat_s2)
            if self.crosscat:
                crosscat_s2 = F.relu(innercat_s2 + self.Crossconv1_stage2(conv3))
            else:
                crosscat_s2 = innercat_s2

        else:

            if self.innercat:
                innercat_s2 = F.relu(self.stage2_pre(cat_s2) + self.Cconv1_stage2(cat_s2))
            else:
                innercat_s2 = F.relu(self.stage2_pre(cat_s2))
            
            if self.crosscat:
                crosscat_s2 = F.relu(self.Mconv2_stage2(innercat_s2) + self.Crossconv1_stage2(conv3))
            else:
                crosscat_s2 = F.relu(self.Mconv2_stage2(innercat_s2))
            
        s2_post = self.stage2_post(crosscat_s2)
        out = self.final_layer(s2_post)
        return out
    
    def init_weights(self):
        print('=> init weights from normal distribution', flush=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)

def get_pose_net(cfg, args, is_train, slim_cfg=None, **kwargs):

    settings = {
        'pre2': [False, False],
        'pre3': [False, False],
        'conv_3_1': [False, False],
        'conv4_stage': [False, False],
        'stage2_pre': [False, False],
        'Cconv1_stage2': [False, False],
        'Mconv2_stage': [True, False],
        'stage2_post': [True, False],
        'innercat': False,
        'crosscat': True,
        'upsample4x': True,
        'sepa': False,
        'up4x_sepa': False
    }

    model = light_model(settings)
    model.init_weights()
    return model