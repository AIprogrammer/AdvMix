import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
import logging


logger = logging.getLogger(__name__)

def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1

def conv_dw_wo_bn(inp, oup, kernel_size, stride, cfg=None):
    if cfg == None:
        return nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        )
    else:
        # how to keep depthwise conv
        return nn.Sequential(
            nn.Conv2d(cfg[0], cfg[0], kernel_size, stride, padding=(kernel_size - 1) // 2, groups=cfg[0], bias=False),
            nn.BatchNorm2d(cfg[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg[0], cfg[2], 1, 1, 0, bias=False)
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
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, gp=False, shuffle=True, cfg=None):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.gp = gp
        self.shuffle = shuffle
        if gp:
            self.conv = conv_dw_wo_bn(inp_dim, out_dim, kernel_size, stride, cfg)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if bn:
            if cfg != None:
                out_dim = cfg[2]
            self.bn = nn.BatchNorm2d(out_dim)
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        # assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn != None:
            x = self.bn(x)
        if self.relu != None:
            x = self.relu(x)
        # add channel shuffle
        group = 2
        if x.data.size()[1] % group == 0 and self.gp and self.shuffle:
            x = channel_shuffle(x, 2)
        # print('========> ', x.shape)
        return x

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class light_model(nn.Module):
    def __init__(self, cfg, settings, slim_cfg=None):
        super(light_model, self).__init__()
        oup_dim = 17

        if slim_cfg == None:
            slim_cfg = [24, 36, 36, 36, 36, 24, 24, 72, 72, 72, 72, 72, 48, 48, 48, 72]

        self.settings = settings
        self.innercat = self.settings['innercat']
        self.crosscat = self.settings['crosscat']
        self.upsample4x = self.settings['upsample4x']

        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        # head block
        self.pre = nn.Sequential(
            Conv(3, slim_cfg[0], 3, 2, bn=True),
            Conv(slim_cfg[0], slim_cfg[1], 3, 2, bn=True),
            Conv(slim_cfg[1], slim_cfg[2], 3, 1, bn=True, gp=settings['pre2'][0], shuffle=settings['pre2'][1]),
            Conv(slim_cfg[2], slim_cfg[3], 3, 1, bn=True, gp=settings['pre3'][0], shuffle=settings['pre3'][1])
        )
        self.conv3_1_stage2 = Conv(slim_cfg[3], slim_cfg[4], 3, 2, bn=True, gp=settings['conv_3_1'][0], shuffle=settings['conv_3_1'][1])

        self.conv4_stage2 = nn.Sequential(
            Conv(slim_cfg[4], slim_cfg[5], 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1]),
            Conv(slim_cfg[5], slim_cfg[6], 3, 1, bn=True, gp=settings['conv4_stage'][0], shuffle=settings['conv4_stage'][1])
        )

        self.stage2_pre_up4x = nn.Sequential(
            Conv(slim_cfg[6], slim_cfg[7], 3, 2, bn=True, ),
            Conv(slim_cfg[7], slim_cfg[8], 3, 1, bn=True, ),
            Conv(slim_cfg[8], slim_cfg[9], 3, 1, bn=True, ),
            Conv(slim_cfg[9], slim_cfg[10], 3, 1, bn=True, ),
            Conv(slim_cfg[10], slim_cfg[11], 3, 1, bn=True, ),
            Conv(slim_cfg[11], 72, 3, 1, bn=False, relu=False,),
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        
        self.stage2_post = nn.Sequential(  
            Conv(72, slim_cfg[12], 1, 1, bn=True),
            Conv(slim_cfg[12], slim_cfg[13], 3, 1, bn=True, gp=settings['stage2_post'][0], shuffle=settings['stage2_post'][1], cfg=[slim_cfg[12], slim_cfg[13], slim_cfg[14]]),
            Conv(slim_cfg[14], slim_cfg[15], 1, 1, bn=True),
            Conv(slim_cfg[15], oup_dim, 1, 1, bn=False, relu=False) # or bilinear downsample
        )

        self.Crossconv1_stage2 = Conv(slim_cfg[3], 72, 1, 1, bn=False, relu=False)

    def forward(self, x: Variable):

        conv3 = self.pre(x)
        conv3_1 = self.conv3_1_stage2(conv3)
        cat_s2 = self.conv4_stage2(conv3_1)

        innercat_s2 = self.stage2_pre_up4x(cat_s2)
        crosscat_s2 = F.relu(innercat_s2 + self.Crossconv1_stage2(conv3))            
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

def get_pose_net(cfg, args, is_train, slim_cfg=None, **kwargs):
    
    'Mconv + stage2_pose 2nd conv + upsample4x'
    settings = {
            'pre2': [False, False],
            'pre3': [False, False],
            'conv_3_1': [False, False],
            'conv4_stage': [False, False],
            'stage2_pre': [False, False],
            'Cconv1_stage2': [False, False],
            # 'stage2_post': [True, False],
            'stage2_post': [False, False],
            'innercat': True,
            'crosscat': True,
            'upsample4x':True
    }
    model = light_model(cfg, settings, slim_cfg)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        try:
            model.init_weights(cfg.MODEL.PRETRAINED)
            model_dict = torch.load('output/coco/baseline/baseline_128_96_Mconv_stage2pose_upsample4x_changesize_woshuffle/model_best_?.pth')
            my_dict = model.state_dict()
            pretrained_weights = {k:v for k, v in model_dict.items() if k in my_dict}
            print('loading from pretrained checkpoint : ', len(pretrained_weights))
            my_dict.update(pretrained_weights)
            model.load_state_dict(my_dict)
        except:
            print('==> checkpoint not exists')
    return model

r"""
'module.stage2_post.0.conv.weight', 'module.stage2_post.0.conv.bias', 'module.stage2_post.0.bn.weight', 'module.stage2_post.0.bn.bias', 'module.stage2_post.0.bn.running_mean', 
'module.stage2_post.0.bn.running_var', 'module.stage2_post.0.bn.num_batches_tracked', 



'module.stage2_post.1.conv.0.weight', 'module.stage2_post.1.conv.1.weight', 'module.stage2_post.1.conv.1.bias', 'module.stage2_post.1.conv.1.running_mean', 
'module.stage2_post.1.conv.1.running_var', 'module.stage2_post.1.conv.1.num_batches_tracked', 'module.stage2_post.1.conv.3.weight', 'module.stage2_post.1.bn.weight', 
'module.stage2_post.1.bn.bias', 'module.stage2_post.1.bn.run
ning_mean', 'module.stage2_post.1.bn.running_var', 'module.stage2_post.1.bn.num_batches_tracked', 



'module.stage2_post.2.conv.weight', 'module.stage2_post.2.conv.bias', 'module.stage2_post.2.bn.weight', 'module.stage2_post.2.bn.bias', 'module.stage2_post.2.bn.running_mean', 
'module.stage2_post.2.bn.running_var', 'module.stage2_post.2.bn.num_batches_tracked', 



'module.stage2_post.3.conv.weight', 'module.stage2_post.3.conv.bias'

"""




