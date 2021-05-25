import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from opt import opt

import os
import logging
import functools


logger = logging.getLogger(__name__)

def createModel(**kw):
    model = light_model()
    return model


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
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


class light_model(nn.Module):
    def __init__(self, cfg):
        super(light_model, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
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
            # Conv(72, 72, 3, 1, bn=True),
            # Conv(72, 72, 3, 1, bn=True),
            # Conv(72, 72, 3, 1, bn=True),
            Conv(72, 72, 3, 1, bn=True),
            Conv(72, 72, 3, 1, bn=False, relu=False),
            # nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(
                    in_channels=72,
                    out_channels=72,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True)
            )
        self.Cconv1_stage2 = Conv(24, 72, 1, 1, bn=False, relu=False)
        # concat self.stage2_pre & self.Cconv1_stage2

        # self.Mconv2_stage2 = nn.Sequential(
        #     Conv(72, 72, 3, 1, bn=False, relu=False),
        #     nn.Upsample(scale_factor=2, mode='bilinear')
        # )
        self.Mconv2_stage2 = nn.ConvTranspose2d(
                    in_channels=72,
                    out_channels=72,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True)
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

def get_pose_net(cfg, is_train, **kwargs):

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    model = light_model(cfg)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


