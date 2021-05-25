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


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class light_model(nn.Module):
    def __init__(self, cfg, args, settings):
        super(light_model, self).__init__()
        # oup_dim = opt.nClasses  #17 for COCO
        '''
        1. change conv3_1 stage
        2. change conv_stage2
        3. change stage2_pre
        4. 
        '''
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
        # head block
        if self.settings['sepa']:
            self.pre = nn.Sequential(
                        Conv(3, 24, 3, 2, bn=True),
                        Conv(24, 36, 3, 2, bn=True),
            )
            self.pre_half_1 = Conv(36, 18, 3, 1, bn=True, gp=settings['pre2'][0], shuffle=settings['pre2'][1])
            self.pre_half_2 = Conv(36, 18, 3, 1, bn=True, gp=settings['pre3'][0], shuffle=settings['pre3'][1])
        
        if self.settings['DFP']:
            
            # self.pre = nn.Sequential(
            #             DFPConv(3, 24, 3, 2, bn=True),
            #             DFPConv(24, 36, 3, 2, bn=True),
            # )
            # self.pre_half_1 = Conv(36, 18, 3, 1, bn=True, gp=settings['pre2'][0], shuffle=settings['pre2'][1])
            # self.pre_half_2 = Conv(36, 18, 3, 1, bn=True, gp=settings['pre3'][0], shuffle=settings['pre3'][1])

            self.pre = nn.Sequential(
                DFPConv(3, 24, 3, 2, bn=True),
                DFPConv(24, 36, 3, 2, bn=True),
                DFPConv(36, 36, 3, 1, bn=True),
                DFPConv(36, 36, 3, 1, bn=True)
            )

        else:
            ### change to dilation conv
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

        # self.stage2_post = nn.Sequential(  
        #     Conv(72, 144, 1, 1, bn=True),
        #     Conv(144, 288, 3, 1, bn=True, gp=settings['stage2_post'][0], shuffle=settings['stage2_post'][1]),
        #     Conv(288, 576, 1, 1, bn=True),
        #     Conv(576, 1152, 1, 1, bn=True),
        #     Conv(1152, 2204, 1, 1, bn=True),
        # )

        ### tmp: unsample4x

        if self.settings['up4x_sepa']:

            self.stage2_pre_up4x = nn.Sequential(
                Conv(24, 72, 3, 2, bn=True,),
            )

            self.stage2_pre_up4x_1 = nn.Sequential(
                Conv(72, 36, 3, 1, bn=True, ),
                Conv(36, 36, 3, 1, bn=True, ),
                Conv(36, 36, 3, 1, bn=True, ),
                Conv(36, 36, 3, 1, bn=True, ),
                Conv(36, 36, 3, 1, bn=False, relu=False,),
                nn.Upsample(scale_factor=4, mode='bilinear')
            )


            self.stage2_pre_up4x_2 = nn.Sequential(
                Conv(72, 36, 3, 1, bn=True, gp=True),
                Conv(36, 36, 3, 1, bn=True, gp=True),
                Conv(36, 36, 3, 1, bn=True, gp=True),
                Conv(36, 36, 3, 1, bn=True, gp=True),
                Conv(36, 36, 3, 1, bn=False,gp=True, relu=False,),
                nn.Upsample(scale_factor=4, mode='bilinear')
            )
        
        else:

            self.stage2_pre_up4x = nn.Sequential(
                Conv(24, 72, 3, 2, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=True, ),
                Conv(72, 72, 3, 1, bn=False, relu=False,),
                nn.Upsample(scale_factor=4, mode='bilinear')
            )

            
            # pass
        # channel should not be same as joint num
        if args.fc_coord:

            # with_r = True
            # rank = 2

            # if with_r:
            #     in_ch = oup_dim + rank + 1
            # else:
            #     in_ch = oup_dim + rank

            # self.coord_conv = coord_conv.AddCoords(rank, with_r, use_cuda=True)
            # self.final_conv = nn.Sequential(  
            #             Conv(in_ch, 12, 1, 1, bn=True, gp=settings['fc_coord'][0], shuffle=settings['fc_coord'][1]),
            #             Conv(12, oup_dim, 1, 1, bn=True, gp=settings['fc_coord'][0], shuffle=settings['fc_coord'][1]),
            # )

            "w/o coord conv"
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.pool = nn.AvgPool2d(2)

            # self.fc = nn.Sequential(
            #         nn.Linear(72, 2 * oup_dim),
            #         nn.BatchNorm1d(2 * oup_dim),
            #         nn.Dropout(0.5),
            #         nn.Tanh() # [-1, 1] speed of exp direct output should not add tanh or relu
            # )

            # self.fc = nn.Sequential(
            #         nn.Linear(2204, 2204),
            #         nn.BatchNorm1d(2204),
            #         nn.Dropout(0.5),
            #         nn.Linear(2204, 2 * oup_dim),
            #         nn.BatchNorm1d(2 * oup_dim),
            #         nn.Dropout(0.5),
            #         # nn.Tanh() # [-1, 1] speed of exp direct output should not add tanh or relu
            # )

            self.fc = nn.Sequential(
                nn.Linear(72, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 4096),
                nn.BatchNorm1d(4096),
                nn.Dropout(0.5),
                nn.Linear(4096, 2 * oup_dim),
                nn.BatchNorm1d(2 * oup_dim),
                nn.Dropout(0.5),
            )

            "fcn"
            # self.fcn = nn.Sequential(
            #     Conv(17,17,3,2,bn=True),
            #     Conv(17,17,3,2,bn=True),
            #     Conv(17,17,3,2,bn=True),
            #     Conv(17,17,3,2,bn=True),
            #     Conv(17,17,3,2,bn=True),
            # )

    def forward(self, x: Variable):
        # if self.settings['sepa']:
        #     x = self.pre(x)
        #     x_1 = self.pre_half_1(x)
        #     x_2 = self.pre_half_1(x)
        #     conv3 = torch.cat((x_1, x_2), dim=1)
        # else:
        #     conv3 = self.pre(x)
        
        # conv3_1 = self.conv3_1_stage2(conv3)

        # cat_s2 = self.conv4_stage2(conv3_1)
    
        # if self.innercat:
        #     innercat_s2 = F.relu(self.stage2_pre(cat_s2) + self.Cconv1_stage2(cat_s2))
        # else:
        #     innercat_s2 = self.stage2_pre(cat_s2)

        # out = self.stage2_post(innercat_s2)


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
            
        out = self.stage2_post(crosscat_s2)

        if self.args.dsntnn:
            # Normalize the heatmaps [0, 1]
            normalized_hm = dsntnn.flat_softmax(out)
            # Calculate the coordinates
            coords = dsntnn.dsnt(normalized_hm)
            # return coords / origin out
            out = [coords, out, normalized_hm]
            # out = coords
        elif self.args.softargmax:
            tmp = out
            coords, maxvals = self.soft_argmax(tmp)
            out = [coords, tmp]
        elif self.args.fc_coord:
            "normalization or not"
            "attention: do not use pooling layer and linear"

            "how to set channels tmp meaningless"
            # tmp = out
            # out = self.coord_conv(out) # +2
            # out = self.final_conv(out)
            # out = self.avg_pool(out)
            # out = out.view(out.shape[0], out.shape[1], -1)
            # out = self.fc(out) # b, 17 ,2
            # out = [out, tmp]

            "w/o coord conv"
            # tmp = crosscat_s2
            # out = self.avg_pool(tmp)
            # out = out.view(out.shape[0], -1) # bct, 17, 16 -- bct, 17, 2
            # out = self.fc(out)
            # # reshape
            # out = out.view(out.shape[0], -1, 2)
            # out = [out, tmp]

            "deep channel"
            # tmp = crosscat_s2
            # out = self.stage2_post(tmp)
            # if out.shape[2] * out.shape[3] > 4:
            #     out = self.pool(out)
            # out = self.avg_pool(out)
            # out = out.view(out.shape[0], -1)

            # out = self.fc(out)
            # out = out.view(out.shape[0], -1, 2)
            # out = [out, tmp]
            r"deep chanel wo changing conv"

            tmp = crosscat_s2 # 72
            out = tmp
            if out.shape[2] * out.shape[3] > 4:
                out = self.pool(out)
            out = self.avg_pool(out)
            out = out.view(out.shape[0], -1)
            out = self.fc(out)
            out = out.view(out.shape[0], -1, 2)
            out = [out, tmp]

            "fcn"
            # tmp = out
            # out = self.fcn(out)
            # print('='*20, out.shape)
            # out = out.view(out.shape[0], out.shape[1], -1)
            # out = [out, tmp]
            "resnet deeppose"
        
        # if softargmax
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
            'sepa': False,
            'up4x_sepa': False,
            'DFP': False,
            'fc_coord':[False, False]
    }

    model = light_model(cfg, args, settings)
    # model = DeepPose(17)
    # model = lightDeepPose(17)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        try:
            model.init_weights(cfg.MODEL.PRETRAINED)
        except:
            pass
        try:
            ### load pretrained weights
            # Mconv weights : 'output/coco/baseline/baseline_128_96_Mconv/model_best.pth'
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

"""setting details

# Mconv
settings = {
            'pre2': [False, False],
            'pre3': [False, False],
            'conv_3_1': [False, False],
            'conv4_stage': [False, False],
            'stage2_pre': [False, False],
            'Cconv1_stage2': [False, False],
            'Mconv2_stage': [True, False],
            'stage2_post': [False, False],
            'innercat': True,
            'crosscat': True,
            'upsample4x':False
    }


# Mconv + stage2_pose 2nd conv
settings = {
            'pre2': [False, False],
            'pre3': [False, False],
            'conv_3_1': [False, False],
            'conv4_stage': [False, False],
            'stage2_pre': [False, False],
            'Cconv1_stage2': [False, False],
            'Mconv2_stage': [True, True],
            'stage2_post': [True, True],
            'innercat': True,
            'crosscat': True,
            'upsample4x':False
    }

"""
