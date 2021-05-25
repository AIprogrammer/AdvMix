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
            Conv(3, 16, 3, 2, bn=True),
            Conv(16, 32, 3, 2, bn=True),
            Conv(32, 32, 3, 1, bn=True),
            Conv(32, 32, 3, 1, bn=True)
        )
        self.conv3_1_stage2 = Conv(32, 32, 3, 2, bn=True)
        self.conv4_stage2 = nn.Sequential(
            Conv(32, 16, 3, 1, bn=True),
            Conv(16, 16, 3, 1, bn=True)
        )

        # body block
        self.stage2_pre = nn.Sequential(
            Conv(16, 48, 3, 2, bn=True),
            Conv(48, 48, 3, 1, bn=True),
            Conv(48, 48, 3, 1, bn=True),
            Conv(48, 48, 3, 1, bn=True),
            Conv(48, 48, 3, 1, bn=True),
            Conv(48, 48, 3, 1, bn=False, relu=False),
            # nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(
                    in_channels=48,
                    out_channels=48,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True)
            )
        self.Cconv1_stage2 = Conv(16, 48, 1, 1, bn=False, relu=False)
        # concat self.stage2_pre & self.Cconv1_stage2

        # self.Mconv2_stage2 = nn.Sequential(
        #     Conv(72, 72, 3, 1, bn=False, relu=False),
        #     nn.Upsample(scale_factor=2, mode='bilinear')
        # )
        self.Mconv2_stage2 = nn.ConvTranspose2d(
                    in_channels=48,
                    out_channels=48,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True)
        self.Crossconv1_stage2 = Conv(32, 48, 1, 1, bn=False, relu=False)
        # concat self.Mconv2_stage2 & self.Crossconv1_stage2

        self.stage2_post = nn.Sequential(  
            Conv(48, 64, 1, 1, bn=True),
            Conv(64, 64, 3, 1, bn=True),
            Conv(64, 64, 1, 1, bn=True),
            Conv(64, oup_dim, 1, 1, bn=False, relu=False) # or bilinear downsample
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


class UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False, with_tanh=True):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if with_tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
    
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, with_tanh=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, with_tanh=with_tanh)  # add the innermost layer
        # unet_block = InnerBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, with_tanh=with_tanh)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        unet_block = UnetBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        unet_block = UnetBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        self.model = UnetBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, with_tanh=with_tanh)  # add the outermost layer
    def forward(self, input):
        """Standard forward"""
        return self.model(input)

def get_pose_net(cfg, is_train, **kwargs):

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    model = light_model(cfg)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


