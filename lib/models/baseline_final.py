import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import dsntnn
import torchvision
# from opt import opt

import os
import logging
from models import coord_conv
from core.inference import get_final_preds, get_final_preds_using_softargmax, SoftArgmax2D



logger = logging.getLogger(__name__)


def conv_dw_wo_bn(inp, oup, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
    )

def conv_dw_w_bn_dilated(inp, oup, kernel_size, stride, dilation=3):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding=dilation, groups=inp, bias=False, dilation=dilation),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
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

class DeepPose(nn.Module):
    """docstring for DeepPose"""
    def __init__(self, nJoints, modelName='resnet50'):
        super(DeepPose, self).__init__()
        self.nJoints = nJoints
        self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
        self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
        self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
        self.feat = nn.Sequential(*list(self.resnet.children())[:-2])
    def forward(self, x):
        tmp = self.feat(x)
        out = self.resnet(x)
        out = out.reshape(-1 , self.nJoints, 2)
        out = [out, tmp]
        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, gp=False, shuffle=False):
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


class lightDeepPose(nn.Module):
    """docstring for DeepPose"""
    def __init__(self, nJoints, modelName='resnet50'):
        super(lightDeepPose, self).__init__()
        self.nJoints = nJoints
        # 144 112 ==> 16 14
        self.feat = nn.Sequential(
                Conv(3, 48, 3, 2, bn=True),
                Conv(48, 128, 3, 2, bn=True),
                Conv(128, 192, 3, 2, bn=True,),
                Conv(192, 192, 3, 1, bn=True,), 
                Conv(192, 192, 3, 1, bn=True,),
                Conv(192, 192, 3, 1, bn=True,),
        )

        self.linear = nn.Sequential(
                    nn.Linear(192, 1024),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 2*nJoints),
                    nn.BatchNorm1d(2*nJoints),
                    nn.Dropout(0.5)
        )  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AvgPool2d(2)  

    def forward(self, x):
        tmp = self.feat(x)
        out = tmp
        if out.shape[2] * out.shape[3] > 4:
            out = self.avg_pool(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.reshape(-1 , self.nJoints, 2)
        out = [out, tmp]
        return out


### remember no shuffle
class DFPConv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, gp=True, shuffle=False):
        super(DFPConv, self).__init__()
        self.inp_dim = inp_dim
        self.gp = gp
        self.shuffle = shuffle
        if gp:
            self.conv1 = conv_dw_w_bn_dilated(inp_dim, out_dim // 2, kernel_size, stride, dilation=1)
            self.conv2 = conv_dw_w_bn_dilated(inp_dim, out_dim // 2, kernel_size, stride, dilation=3)
            self.joint = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1)
        else:
            ### to do
            self.conv1 = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
            self.conv2 = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
            self.joint = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)

        self.relu = None
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        ### fuse the diation layer
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.joint(torch.cat((x1, x2), dim=1))

        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        # add channel shuffle
        group = 2
        if x.data.size()[1] % group == 0 and self.gp and self.shuffle:
            x = channel_shuffle(x, 2)
        return x

class BasicBlock3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, gp=True):
        super(BasicBlock3x3, self).__init__()
        if gp:
            self.conv1 = Conv(in_planes, out_planes, kernel_size=3, stride=stride, gp=gp, bn=False, relu=False, shuffle=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn1 = torch.nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicBlock1x1(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1):
        super(BasicBlock1x1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, dilation=dilation)
        self.bn1 = torch.nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, in_planes, stride=1, dilation=[1,1,1,1], downsample=None):
        super(Bottleneck, self).__init__()
        out_planes = in_planes // 2
        self.block1 = BasicBlock1x1(in_planes, out_planes, dilation=dilation[0])
        self.block2 = BasicBlock3x3(out_planes, out_planes // 2, dilation=dilation[1])
        self.block3 = BasicBlock3x3(out_planes // 2, out_planes // 4, dilation=dilation[2])
        self.block4 = BasicBlock3x3(out_planes // 4, out_planes // 4, dilation=dilation[3])
        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)

        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        out = torch.cat([out1, out2, out3, out4], 1)
        return out

def make_layers(n_layers, in_planes):
    layers = []
    for i in range(n_layers):
        layers.append(Bottleneck(in_planes))
    
    return layers


class light_model(nn.Module):
    def __init__(self, cfg, args, settings):
        super(light_model, self).__init__()
        oup_dim = 17

        self.args = args
        self.settings = settings
        self.innercat = self.settings['innercat']
        self.crosscat = self.settings['crosscat']
        self.upsample4x = self.settings['upsample4x']

        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        if self.args.softargmax:
            self.soft_argmax = SoftArgmax2D(cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0], beta=160)

        if self.settings['sepa']:
            self.pre = nn.Sequential(
                        Conv(3, 24, 3, 2, bn=True),
                        Conv(24, 36, 3, 2, bn=True),
            )
            self.pre_half_1 = Conv(36, 24, 3, 1, bn=True, gp=False, shuffle=False)
            self.pre_half_2 = Conv(36, 12, 3, 1, bn=True, gp=True, shuffle=False)
            self.pre_final = Conv(36, 36, 3, 1, bn=True, gp=False, shuffle=False)
        else:
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

        self.Crossconv1_stage2 = Conv(36, 72, 1, 1, bn=False, relu=False)

        self.stage2_post = nn.Sequential(  
            Conv(72, 48, 1, 1, bn=True),
            Conv(48, 48, 3, 1, bn=True, gp=settings['stage2_post'][0], shuffle=settings['stage2_post'][1]),
            Conv(48, 72, 1, 1, bn=True),
            Conv(72, oup_dim, 1, 1, bn=False, relu=False) # or bilinear downsample
        )

        if self.settings['up4x_sepa']:

            self.stage2_pre_up4x = nn.Sequential(
                Conv(24, 72, 3, 2, bn=True,),
            )

            self.stage2_pre_up4x_1 = nn.Sequential(
                Conv(72, 36, 3, 1, bn=True, ),
                Conv(36, 36, 3, 1, bn=True, ),
                Conv(36, 36, 3, 1, bn=True, ),
            )

            self.stage2_pre_up4x_2 = nn.Sequential(
                Conv(72, 36, 3, 1, bn=True, gp=True),
                Conv(36, 36, 3, 1, bn=True, gp=True),
                Conv(36, 36, 3, 1, bn=True, gp=True),
            )
            self.stage2_pre_up4x_final = nn.Sequential(
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, relu=False,),

            )
            self.upsample4x_func = nn.Upsample(scale_factor=4, mode='bilinear')

        # if self.settings['up4x_sepa']:

        #     self.stage2_pre_up4x = nn.Sequential(
        #         Conv(24, 72, 3, 2, bn=True,),
        #         Conv(72, 36, 1, 1, bn=True, ),
        #         Conv(36, 36, 1, 1, bn=True, ),
        #     )

        #     self.stage2_pre_up4x_1 = nn.Sequential(
        #         Conv(36, 36, 3, 1, bn=True, ),
        #         Conv(36, 36, 3, 1, bn=True, ),
        #         Conv(36, 36, 1, 1, bn=True, ),
        #     )

        #     self.stage2_pre_up4x_2 = nn.Sequential(
        #         Conv(36, 36, 3, 1, bn=True, gp=True),
        #         Conv(36, 36, 3, 1, bn=True, gp=True),
        #         Conv(36, 36, 1, 1, bn=True, ),
        #     )
        #     self.stage2_pre_up4x_final = nn.Sequential(
        #         Conv(72, 72, 1, 1, bn=True, ),
        #         Conv(72, 72, 3, 1, bn=True, relu=False,),

        #     )
        #     self.upsample4x_func = nn.Upsample(scale_factor=4, mode='bilinear')

        # if self.settings['up4x_sepa']:
        #     layers = [Conv(24, 72, 3, 2, bn=True,)]
        #     layers += make_layers(2, 72)
        #     layers += [Conv(72, 72, 3, 1, bn=True, relu=False,),                 
        #                 nn.Upsample(scale_factor=4, mode='bilinear')
        #             ]
        #     self.stage2_pre_up4x = nn.Sequential(*layers)
        else:

            self.stage2_pre_up4x = nn.Sequential(
                Conv(24, 72, 3, 2, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, relu=False,),
                nn.Upsample(scale_factor=4, mode='bilinear')
            )

    def forward(self, x: Variable):
        # pre edit
        if self.settings['sepa']:
            x = self.pre(x)
            x_1 = self.pre_half_1(x)
            x_2 = self.pre_half_2(x)
            tmp_conv3 = torch.cat((x_1, x_2), dim=1)
            conv3 = self.pre_final(tmp_conv3)
        else:
            conv3 = self.pre(x)
        
        conv3_1 = self.conv3_1_stage2(conv3)
        cat_s2 = self.conv4_stage2(conv3_1)

        # sepa
        if self.settings['up4x_sepa']:
            # mid4x = self.stage2_pre_up4x(cat_s2)
            # mid4x_1 = self.stage2_pre_up4x_1(mid4x)
            # mid4x_2 = self.stage2_pre_up4x_2(mid4x)
            # tmp_innercat_s2 = self.stage2_pre_up4x_final(torch.cat((mid4x_1, mid4x_2), dim=1))
            # innercat_s2 = self.upsample4x_func(tmp_innercat_s2)
            innercat_s2 = self.stage2_pre_up4x(cat_s2)
        else:
            innercat_s2 = self.stage2_pre_up4x(cat_s2)
        
        crosscat_s2 = F.relu(innercat_s2 + self.Crossconv1_stage2(conv3))
        out = self.stage2_post(crosscat_s2)

        return out
    
    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconvconvert_locations_to_boxes weights from normal distribution')
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
            for m in self.Convfinal_layer.modules():
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

def get_pose_net(cfg, args, is_train, **kwargs):
    
    'Mconv + stage2_pose 2nd conv + upsample4x'
    'FOR CHANGE SIZES : remember to change size in .yaml'
    'FOR CHANGING CAHNNELS : remember to change num of channels base on baseline_size_channel.py'
    settings = {
            'pre2': [False, False],
            'pre3': [False, False],
            'conv_3_1': [False, False],
            'conv4_stage': [False, False],
            'stage2_pre': [False, False],
            'Cconv1_stage2': [False, False],
            'Mconv2_stage': [True, False],
            'stage2_post': [True, False],
            'innercat': True,
            'crosscat': True,
            'upsample4x':True,
            'sepa': True,
            'up4x_sepa': False,
            'DFP': False,
            'fc_coord':[False, False]
    }

    model = light_model(cfg, args, settings)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        try:
            model.init_weights(cfg.MODEL.PRETRAINED)
        except:
            pass
        try:
            model_dict = torch.load('output/coco/baseline/baseline_128_96_Mconv_stage2pose_upsample4x_changesize_woshuffle/model_best.pth')
            my_dict = model.state_dict()
            pretrained_weights = {k:v for k, v in model_dict.items() if k in my_dict}
            print('loading from pretrained checkpoint : ', len(pretrained_weights))
            my_dict.update(pretrained_weights)
            # not load pretrained model for fair comparison
            # model.load_state_dict(my_dict)
        except:
            print('the resume not exists')

    return model