import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import os
import logging
import functools

logger = logging.getLogger(__name__)

def conv_dw_wo_bn(inp, oup, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
    )

class BasicBlock3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1):
        super(BasicBlock3x3, self).__init__()
        # if dilation == 1:
        #     self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # elif dilation == 2:
        #     self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class BasicBlock1x1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicBlock1x1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_planes, stride=1, dilation=1, keep=False, downsample=None):
        super(Bottleneck, self).__init__()
        if keep:
            out_planes = in_planes
        else:
            out_planes = in_planes // 2
        self.block1 = BasicBlock1x1(in_planes, out_planes)
        self.block2 = BasicBlock3x3(out_planes, out_planes // 2)
        self.block3 = BasicBlock3x3(out_planes // 2, out_planes // 4)
        self.block4 = BasicBlock3x3(out_planes // 4, out_planes // 4)
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

class DownConv(nn.Module):
    
    def __init__(self, in_ch, out_ch, padding_type='reflect', lastdown=False, resblock=True):
        super(DownConv, self).__init__()
        conv_block = []
        if lastdown:

            conv_block += [Conv(in_ch, out_ch, 3, 2,),
                        Conv(out_ch, out_ch, 3, 1,)
            ]

        else:

            conv_block += [Conv(in_ch, out_ch, 3, 2),
                        Conv(out_ch, out_ch, 3, 1,)
            ]
        
        self.downconv = nn.Sequential(*conv_block)
    
    def forward(self, x):
        x = self.downconv(x)
        return x

class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch, outermost=False, bilinear=False, use_bias=True, resblock=True):
    
        super(UpConv, self).__init__()
        if bilinear:
            self.upconv = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        else:
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        
        if outermost:
            self.upconv = [nn.ReLU()]
        else:
            self.upconv += [nn.ReLU(), nn.BatchNorm2d(out_ch)]
        
        self.upconv = nn.Sequential(*self.upconv) 
        self.com = Conv(out_ch * 2, out_ch * 2, 3, 1)
    
    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.com(x)
        return x

class Out(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=False):
        super(Out, self).__init__()
        self.conv = nn.Sequential(
                        nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(in_ch),
                        nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0

                        )
        )
    def forward(self, x):
        return self.conv(x)

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=False):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(
                        Conv(in_ch, out_ch, 3, 2, bn=True),
                        Conv(out_ch, out_ch, 3, 1, bn=True),
                        Conv(out_ch, out_ch, 3, 1, bn=True)
        )
    
    def forward(self, x):
        return self.conv(x)


### skip connection ==> deconv ==> final layer 
class Unet(nn.Module):
    def __init__(self, cfg, n_channels, n_classes, layers=(2, 3, 4, 1), block=Bottleneck):
        super(Unet, self).__init__()
        
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        self.relu = nn.ReLU(inplace=True)
        self.inc = InConv(n_channels, 32)
        
        self.down1 = self._make_layer(32, 64, block, layers[0], keep=True)
        self.down2 = self._make_layer(64, 64, block, layers[1], keep=True)
        self.down3 = self._make_layer(64, 64, block, layers[2], keep=True)
        self.down4 = self._make_layer(64, 64, block, layers[3], keep=True)

        self.up1 = UpConv(64, 64, bilinear=False)
        self.up2 = UpConv(128, 64, bilinear=False)
        self.up3 = Out(128, n_classes)
    
    def _make_layer(self, in_planes, out_planes, block, blocks, keep=True):
        layers = []
        downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                                   torch.nn.BatchNorm2d(out_planes), self.relu,
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        in_planes = out_planes
        layers.append(block(in_planes, downsample=downsample))
        
        if keep:
            in_planes = in_planes
        else:
            in_planes = in_planes * 2
        
        for i in range(1, blocks):
            layers.append(block(in_planes, downsample=None))
        return nn.Sequential(*layers)



        
    def forward(self, x):
        x1 = self.inc(x) # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 32
        x4 = self.down3(x3) # 32
        x5 = self.down4(x4) # 32
        
        '''
        torch.Size([1, 16, 64, 48])
        torch.Size([1, 32, 32, 24])
        torch.Size([1, 32, 16, 12])
        torch.Size([1, 32, 8, 6])
        torch.Size([1, 32, 4, 3])
        '''
        u1 = self.up1(x5,x4)
        u2 = self.up2(u1,x3)
        u3 = self.up3(u2)  # u2: 64x16x12

        # return [x1,x2,x3,x4,x5,u1,u2,u3]
        return u3
    
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


def get_pose_net(cfg, args, is_train, **kwargs):

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    model = Unet(cfg, 3, 17)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model

