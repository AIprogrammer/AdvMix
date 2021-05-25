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




class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class light_model(nn.Module):
    def __init__(self, cfg, args, settings):
        super(light_model, self).__init__()
        self.args = args
        oup_dim = 17

        self.settings = settings
        self.innercat = self.settings['innercat']
        self.crosscat = self.settings['crosscat']
        self.upsample4x = self.settings['upsample4x']
        self.deconv_with_bias = False


        self.pool = nn.MaxPool2d(kernel_size=2)
        in_ch = 3

        if self.settings['coord']:
            with_r = True
            rank = 2
            if with_r:
                in_ch = in_ch + rank + 1
            else:
                in_ch = in_ch + rank
            self.coord_conv = coord_conv.AddCoords(rank, with_r, use_cuda=True)
        
        self.features = nn.Sequential(
            Conv(in_ch, 8, 5, 2,bn=True),
            self.pool,
            Conv(8, 32, 3, 1, bn=True),
            Conv(32, 32, 3, 1, bn=True),
            self.pool,
            Conv(32, 64, 3, 1, bn=True),
            Conv(64, 64, 3, 1, bn=True),
            self.pool,
            Conv(64, 128, 3, 1, bn=True),
            Conv(128,128, 3, 1, bn=True),
            Conv(128, 64, 3, 1, bn=True),
            Conv(64, 64, 3, 1, bn=True),
            # Conv(64, 17, 3, 1, bn=True),
        )

        # 
        if args.fc_coord:

            "w/o coord conv"
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

            self.fc = nn.Sequential(
                nn.Linear(64, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 4096),
                nn.BatchNorm1d(4096),
                nn.Dropout(0.5),
                nn.Linear(4096, 2 * oup_dim),
                nn.BatchNorm1d(2 * oup_dim),
                nn.Dropout(0.5),
            )


            # self.fc = nn.Sequential(
            #     nn.Linear(48, 128),
            #     nn.BatchNorm1d(17),
            #     nn.Dropout(0.5),
            #     nn.Linear(128, 64),
            #     nn.BatchNorm1d(17),
            #     nn.Dropout(0.5),
            #     nn.Linear(64, 2),
            #     nn.BatchNorm1d(17),
            #     nn.Dropout(0.5),
            # )
        self.init_weights()
    def forward(self, x: Variable):
        if self.settings['coord']:
            x = self.coord_conv(x)
        
        tmp = self.features(x)

        if self.args.dsntnn:
            # Normalize the heatmaps [0, 1]
            normalized_hm = dsntnn.flat_softmax(out)
            # Calculate the coordinates
            coords = dsntnn.dsnt(normalized_hm)
            # return coords / origin out
            out = [coords, out, normalized_hm]
        
        if self.args.fc_coord:
            "normalization or not"
            # tmp = out
            # out = self.coord_conv(out) # +2
            # out = self.final_conv(out)
            # out = self.avg_pool(out)
            # out = out.view(out.shape[0], out.shape[1], -1)
            # out = self.fc(out) # b, 17 ,2
            # out = [out, tmp]

            r"deep chanel wo changing conv"
            out = tmp
            if out.shape[2] * out.shape[3] > 4:
                out = self.pool(out)
            out = self.avg_pool(out)
            # out = out.view(out.shape[0], out.shape[1], -1)
            out = out.view(out.shape[0], -1)
            print('==> debug', out.shape)
            out = self.fc(out)
            out = out.view(out.shape[0], -1, 2)
            out = [out, tmp]

        return out
    
    def init_weights(self, pretrained=''):
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
        'up4x_sepa': False,
        'coord': True,
        'fc_coord': [False, False]
    }
    
    # model = light_model(cfg, args, settings)
    model = DeepPose(17)
    return model


if __name__ == '__main__':
    rank = 2
    with_r = True

    coord_conv = coord_conv.AddCoords(rank, with_r, use_cuda=True)
    final_conv = nn.Sequential(  
                Conv(3, 12, 1, 1, bn=True),
                Conv(12, 17, 1, 1, bn=True),
    )

    print(coord_conv(torch.rand(1,3,128,96)).shape)