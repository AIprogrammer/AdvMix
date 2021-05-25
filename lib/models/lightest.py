import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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

        if self.settings['sepa']:
            self.pre = nn.Sequential(
                        Conv(3, 12, 3, 2, bn=True),
                        Conv(12, 24, 3, 2, bn=True),
            )
            self.pre_half_1 = Conv(24, 12, 3, 1, bn=True, gp=False, shuffle=False)
            self.pre_half_2 = Conv(24, 12, 3, 1, bn=True, gp=True, shuffle=False)
            self.pre_final = Conv(24, 24, 3, 1, bn=True, gp=False, shuffle=False)
        
        else:
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

        # conv
        self.stage2_pre_up4x = nn.Sequential(
        Conv(24, 32, 3, 2, bn=True, ),
        Conv(32, 32, 3, 1, bn=True, ),
        Conv(32, 32, 3, 1, bn=True, ),
        Conv(32, 32, 3, 1, bn=True, ),
        Conv(32, 32, 3, 1, bn=True, ),
        Conv(32, 32, 3, 1, bn=True, relu=False, ),
        nn.Upsample(scale_factor=4, mode='bilinear'),
        Conv(32, 32, 1, 1, bn=True, relu=False, ),
        )

        self.Crossconv1_stage2 = Conv(24, 32, 1, 1, bn=True, relu=False)

        self.stage2_post = nn.Sequential(  
            Conv(64, 32, 1, 1, bn=True),
            Conv(32, 32, 3, 1, bn=True, gp=settings['stage2_post'][0], shuffle=settings['stage2_post'][1]),
            Conv(32, 64, 1, 1, bn=True)
        )

        self.final_layer = Conv(64, oup_dim, 1, 1, bn=False, relu=False)  # or bilinear downsample

        self.init_weights()

    def forward(self, x: Variable):

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
        innercat_s2 = self.stage2_pre_up4x(cat_s2)
        crosscat_s2 = F.relu(torch.cat((innercat_s2, self.Crossconv1_stage2(conv3)), dim=1))

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
        'up4x_sepa': False
    }
    
    model = light_model(settings)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights()
    return model