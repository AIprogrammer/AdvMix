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
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


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
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

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
    def __init__(self, cfg, settings):
        super(light_model, self).__init__()
        # oup_dim = opt.nClasses  #17 for COCO
        '''
        1. change conv3_1 stage
        2. change conv_stage2
        3. change stage2_pre
        4. 
        '''
        oup_dim = 17

        self.settings = settings
        self.innercat = self.settings['innercat']
        self.crosscat = self.settings['crosscat']

        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        # head block
        self.pre = nn.Sequential(
            Conv(3, 24, 3, 2, bn=True),
            Conv(24, 36, 3, 2, bn=True),
            Conv(36, 36, 3, 1, bn=True, gp=settings['pre2'][0], shuffle=settings['pre2'][1]),
            Conv(36, 36, 3, 1, bn=True, gp=settings['pre3'][0], shuffle=settings['pre3'][1])
        )
        self.conv3_1_stage2 = Conv(36, 36, 3, 2, bn=True, gp=settings['conv_3_1'][0], shuffle=settings['conv_3_1'][1])
        self.conv4_stage2 = nn.Sequential(
            Conv(36, 24, 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1]),
            Conv(24, 24, 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1])
        )

        # body block
        self.stage2_pre = nn.Sequential(
            Conv(24, 72, 3, 2, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=False, relu=False,),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.Cconv1_stage2 = Conv(24, 72, 1, 1, bn=False, relu=False)
        # concat self.stage2_pre & self.Cconv1_stage2

        self.Mconv2_stage2 = nn.Sequential(
            Conv(72, 72, 3, 1, bn=False, relu=False, gp=settings['Mconv2_stage'][0], shuffle=settings['Mconv2_stage'][1]),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.Crossconv1_stage2 = Conv(36, 72, 1, 1, bn=False, relu=False)
        # concat self.Mconv2_stage2 & self.Crossconv1_stage2

        self.stage2_post = nn.Sequential(  
            Conv(72, 48, 1, 1, bn=True),
            Conv(48, 48, 3, 1, bn=True, gp=settings['stage2_post'][0], shuffle=settings['stage2_post'][1]),
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


class light_model_wo_shotcut(nn.Module):
    def __init__(self, cfg, settings):
        super(light_model_wo_shotcut, self).__init__()
        # oup_dim = opt.nClasses  #17 for COCO
        oup_dim = 17

        self.setting = settings
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        # head block
        self.pre = nn.Sequential(
            Conv(3, 24, 3, 2, bn=True),
            Conv(24, 36, 3, 2, bn=True),
            Conv(36, 36, 3, 1, bn=True, gp=settings['pre2'][0], shuffle=settings['pre2'][1]),
            Conv(36, 36, 3, 1, bn=True, gp=settings['pre3'][0], shuffle=settings['pre3'][1])
        )
        self.conv3_1_stage2 = Conv(36, 36, 3, 2, bn=True, gp=settings['conv_3_1'][0], shuffle=settings['conv_3_1'][1])
        self.conv4_stage2 = nn.Sequential(
            Conv(36, 24, 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1]),
            Conv(24, 24, 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1])
        )

        # body block
        self.stage2_pre = nn.Sequential(
            Conv(24, 72, 3, 2, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=True, ),
            Conv(72, 72, 3, 1, bn=False, relu=False,),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.Cconv1_stage2 = Conv(24, 72, 1, 1, bn=False, relu=False)
        # concat self.stage2_pre & self.Cconv1_stage2

        self.Mconv2_stage2 = nn.Sequential(
            Conv(72, 72, 3, 1, bn=False, relu=False, gp=settings['Mconv2_stage'][0], shuffle=settings['Mconv2_stage'][1]),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.Crossconv1_stage2 = Conv(36, 72, 1, 1, bn=False, relu=False)
        # concat self.Mconv2_stage2 & self.Crossconv1_stage2

        self.stage2_post = nn.Sequential(  
            Conv(72, 48, 1, 1, bn=True),
            Conv(48, 48, 3, 1, bn=True, gp=settings['stage2_post'][0], shuffle=settings['stage2_post'][1]),
            Conv(48, 72, 1, 1, bn=True),
            Conv(72, oup_dim, 1, 1, bn=False, relu=False) # or bilinear downsample
        )

    def forward(self, x: Variable):

        conv3 = self.pre(x)
        conv3_1 = self.conv3_1_stage2(conv3)

        cat_s2 = self.conv4_stage2(conv3_1)

        if self.innercat:
            innercat_s2 = F.relu(self.stage2_pre(cat_s2) + self.Cconv1_stage2(cat_s2))
        else:
            innercat_s2 = F.relu(self.stage2_pre(cat_s2))

        if self.crosscat:
            crosscat_s2 = F.relu(self.Mconv2_stage2(innercat_s2) + self.Crossconv1_stage2(conv3))
        else:
            crosscat_s2 = F.relu(self.Mconv2_stage2(innercat_s2))

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

    settings = {
            'pre2': [False, False],
            'pre3': [False, False],
            'conv_3_1': [False, False],
            'conv4_stage': [False, False],
            'stage2_pre': [False, False],
            'Cconv1_stage2': [False, False],
            'Mconv2_stage': [True, True],
            'stage2_post': [False, False],
            'innercat': True,
            'crosscat': False
    }

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    model = light_model(cfg, settings)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    
        ### load pretrained weights
        # Mconv weights : 'output/coco/baseline/baseline_128_96_Mconv/model_best.pth'
        model_dict = torch.load('output/coco/baseline/baseline_128_96_Mconv/model_best.pth')
        my_dict = model.state_dict()
        pretrained_weights = {k:v for k, v in model_dict.items() if k in my_dict}
        print('loading from pretrained checkpoint : ', len(pretrained_weights))
        my_dict.update(pretrained_weights)

    return model


