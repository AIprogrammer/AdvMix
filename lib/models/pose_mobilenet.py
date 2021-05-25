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

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from opt import opt

import os
import logging



logger = logging.getLogger(__name__)

def createModel(**kw):
    model = light_model()
    return model

def hcf(a,b):
    if a<b: 
        a,b = b,a
    r = a % b 
    while r != 0:
        a,b = b,r
        r = a % b
    return b
"""

File includes:
1. baseline + group conv
2. simple mobilenet total
3. remember deconv changes: not related to baseline 

"""


# groups = hcf(inp_dim, out_dim) if kernel_size != 1 else 1
# assert inp_dim % groups == 0 and out_dim % groups == 0, 'input dim {} can not be divided by {}'.format(inp_dim, groups)

def conv_dw_wo_bn(inp, oup, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
    )

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, groups=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        # self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.conv = conv_dw_wo_bn(inp_dim, out_dim, kernel_size, stride)
        
        self.relu = None
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class light_model_mobile(nn.Module):
    def __init__(self):
        super(light_model_mobile, self).__init__()
        # oup_dim = opt.nClasses  #17 for COCO
        oup_dim = 17
        # head block
        self.pre = nn.Sequential(
            Conv(3, 24, 3, 2, bn=True),
            Conv(24, 36, 3, 2, bn=True),
            Conv(36, 36, 3, 1, bn=True),
            Conv(36, 36, 3, 1, bn=True)
        )
        self.conv3_1_stage2 = Conv(36, 36, 3, 2, bn=True)
        self.conv4_stage2 = nn.Sequential(
            Conv(36, 24, 3, 1, bn=True),
            Conv(24, 24, 3, 1, bn=True)
        )

        # body block
        self.stage2_pre = nn.Sequential(
            Conv(24, 72, 3, 2, bn=True),
            Conv(72, 72, 3, 1, bn=True),
            Conv(72, 72, 3, 1, bn=True),
            Conv(72, 72, 3, 1, bn=True),
            Conv(72, 72, 3, 1, bn=True),
            Conv(72, 72, 3, 1, bn=False, relu=False),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.Cconv1_stage2 = Conv(24, 72, 1, 1, bn=False, relu=False)
        # concat self.stage2_pre & self.Cconv1_stage2

        self.Mconv2_stage2 = nn.Sequential(
            Conv(72, 72, 3, 1, bn=False, relu=False),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.Crossconv1_stage2 = Conv(36, 72, 1, 1, bn=False, relu=False)
        # concat self.Mconv2_stage2 & self.Crossconv1_stage2

        self.stage2_post = nn.Sequential(  
            Conv(72, 48, 1, 1, bn=True),
            Conv(48, 48, 3, 1, bn=True),
            Conv(48, 72, 1, 1, bn=True),
            Conv(72, oup_dim, 1, 1, bn=False, relu=False) # or bilinear downsample
        )

    def forward(self, x: Variable):

        conv3 = self.pre(x)
        conv3_1 = self.conv3_1_stage2(conv3)

        cat_s2 = self.conv4_stage2(conv3_1)
        innercat_s2 = F.relu(self.stage2_pre(cat_s2) + self.Cconv1_stage2(cat_s2))
        crosscat_s2 = F.relu(self.Mconv2_stage2(innercat_s2) + self.Crossconv1_stage2(conv3))

        out = self.stage2_post(crosscat_s2)
        return out
    
    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
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


### official mobilenet : groups == inp_dim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class PoseMobileNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64 
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(PoseMobileNet, self).__init__()

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer = self._make_layer()

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )


    def inner_conv_bn(self, inp, oup, stride, bias=False):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=bias),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def inner_conv_dw(self, inp, oup, stride=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=bias),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    
    def _make_layer(self,):
        self.mobile_model = nn.Sequential(
            self.inner_conv_bn(     3,  32, stride=2, bias=False),
            self.inner_conv_dw(     32,  4, stride=1, bias=False),
            self.inner_conv_dw(     4,  4, stride=1, bias=False),
            self.inner_conv_dw(     4,  8, stride=2, bias=False),
            self.inner_conv_dw(     8,  8, stride=1, bias=False),
            self.inner_conv_dw(     8,  16, stride=2, bias=False),
            self.inner_conv_dw(     16,  16, stride=1, bias=False),
            self.inner_conv_dw(     16,  32, stride=2, bias=False),
            self.inner_conv_dw(     32,  32, stride=1, bias=False),
            self.inner_conv_dw(     32,  64, stride=2, bias=False),
            self.inner_conv_dw(     64,  64, stride=1, bias=False),
            self.inner_conv_dw(     64,  64, stride=1, bias=False),

            # self.inner_conv_dw( 64, 128, stride=2),
            # self.inner_conv_dw(128, 128),
            # self.inner_conv_dw(128, 256, stride=2),
            # self.inner_conv_dw(256, 256),
            # self.inner_conv_dw(256, 512),  # conv4_2
            # self.inner_conv_dw(512, 512),
            # self.inner_conv_dw(512, 512),
            # self.inner_conv_dw(512, 512),
            # self.inner_conv_dw(512, 512),
            # self.inner_conv_dw(512, 512)   # conv5_5
        )
        return self.mobile_model

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
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


class MobilePretrain(nn.Module):

    def __init__(self, cfg, mode='small',width_mult=1.0, feat_weight=''):

        self.inplanes = 96
        input_channel = 16
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(MobilePretrain, self).__init__()

        from models.mobilenetv3 import make_divisible, MobileBottleneck, conv_bn, Hswish

        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError
        
        self.mob_conv = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.mob_conv.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        if os.path.exists(feat_weight):
            model_state_dict = torch.load(feat_weight)

        self.features = nn.Sequential(*self.mob_conv)
        
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def forward(self, x):

        feat = self.features(x)
        # print('='*20, feat.shape)
        deconv = self.deconv_layers(feat)
        out = self.final_layer(deconv)

        return out


    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

class ShufflePretrain(nn.Module):

    def __init__(self, cfg, width_mult=1.,):
        super(ShufflePretrain, self).__init__()
        if width_mult == 1:
            self.inplanes = 464
        else:
            self.inplanes = 192

        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        # assert input_size % 32 == 0
        from models.shufflenetv2 import InvertedResidual, conv_bn

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(''))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
	            #inp, oup, stride, benchmodel:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)


        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        feat = self.features(x)
        # print('='*20, feat.shape)
        deconv = self.deconv_layers(feat)
        out = self.final_layer(deconv)

        return out
        

class EfficientPose(BaseModel):

    def __init__(self, cfg, is_train):
        super(EfficientPose, self).__init__(cfg)
        self.inplanes = 1280
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        # from model.efficient import EfficientNet
        # self.model = EfficientNet.from_pretrained("efficientnet-b0")
        if is_train:
            from efficientnet_pytorch import EfficientNet
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            from efficientnet_pytorch import EfficientNet
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
        

        # self.head = nn.Sequential(*list(self.model.children())[:2])
        # self.feat = list(self.model.children())[2]
        # self.mid_layers = Conv(1280, 32, kernel_size=3, stride=1)
        
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        # self.deconv_layers = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear'),
        #                 nn.Upsample(scale_factor=2, mode='bilinear'),
        #                 nn.Upsample(scale_factor=2, mode='bilinear')
        # )

        
        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        # self.final_layer = Conv(extra.NUM_DECONV_FILTERS[-1], cfg.MODEL.NUM_JOINTS,
        #                         extra.FINAL_CONV_KERNEL, 1)

        ### for onnx
        self.model.set_swish(memory_efficient=False)

    def forward(self, x):
        x = self.model.extract_features(x)
        # x = self.head(x)
        # for sub in self.feat[:12]:
        #     x = sub(x)
        # x = self.mid_layers(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

def get_pose_net(cfg, is_train, **kwargs):

    '''
    load light_model
    load posemobile model
    '''
    # num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    # model = light_model_mobile()
    # model = PoseMobileNet(cfg, **kwargs)
    # if is_train and cfg.MODEL.INIT_WEIGHTS:
        # model.init_weights(cfg.MODEL.PRETRAINED)


    '''
    load mobilenet v3 weights
    '''
    # feat_weight = 'pretrained/mobilenetv3_small_67.4.pth.tar'
    # model = MobilePretrain(cfg)
    # pre_mobile_dict = torch.load(feat_weight)
    # my_dict = model.state_dict()
    # pretrained_dict = {k:v for k,v in pre_mobile_dict.items() if k in my_dict}
    # my_dict.update(pretrained_dict)
    # model.load_state_dict(my_dict)

    '''
    load shufflenet v2 pretrained weights
    '''
    'multi=1'
    # feat_weight = 'pretrained/shufflenetv2_x1_69.402_88.374.pth.tar'
    # model = ShufflePretrain(cfg)
    # # 'multi=0.5'
    # # feat_weight = 'pretrained/shufflenetv2_x0.5_60.646_81.696.pth.tar'
    # # model = ShufflePretrain(cfg, width_mult=0.5)
    # pre_mobile_dict = torch.load(feat_weight)
    # my_dict = model.state_dict()
    # pretrained_dict = {k:v for k,v in pre_mobile_dict.items() if k in my_dict}
    # print('loading pretrained weights from {}, length dict {}'.format(feat_weight, len(pretrained_dict)))
    # my_dict.update(pretrained_dict)
    # model.load_state_dict(my_dict)

    '''
    efficient
    '''
    model = EfficientPose(cfg, is_train=is_train)
    
    return model

if __name__ == '__main__':
    pass

