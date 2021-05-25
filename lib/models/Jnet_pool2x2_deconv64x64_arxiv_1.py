import os
# from extensions.AE.AE_loss import AEloss
import torch
import torch.nn as nn
#from task.loss import HeatmapLoss, tagLoss
def createModel(**kw):
    model = PoseNet()
    return model

def conv_dw_wo_bn(inp, oup, kernel_size, stride, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding=dilation, groups=inp, bias=False, dilation=dilation),
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

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, gp=False, shuffle=False, bias=False, dilation=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.gp = gp
        self.shuffle = shuffle
        if gp:
            self.conv = conv_dw_wo_bn(inp_dim, out_dim, kernel_size, stride, dilation=dilation)
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

class BasicBlock3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, gp=True):
        super(BasicBlock3x3, self).__init__()
        if gp:
            self.conv1 = Conv(in_planes, out_planes, kernel_size=3, stride=stride, bias=False, gp=gp, dilation=dilation)
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

class PoseNet(nn.Module):
    # (2, 3, 4, 1)
    def __init__(self, block=Bottleneck, layers=(1, 2, 3, 1), nstack=1, inp_dim=256, oup_dim=17, num_parts=17):
        super(PoseNet, self).__init__()
        self.num_parts = num_parts
        base_channels = 32
        self.inplanes = base_channels
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1])
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2])

        # deconv params
        self.num_deconv_layers = 2
        self.num_deconv_filters = [64, 64]
        self.num_deconv_kernels = [4, 4]    
        self.deconv_with_bias = False

        # used for deconv layers
        # self.deconv_layers = self._make_deconv_layer(
        #     self.num_deconv_layers,
        #     self.num_deconv_filters,
        #     self.num_deconv_kernels
        # )

        self.deconv_layers = self._upsample_layer(self.num_deconv_filters, 2)

        self.final_layer = nn.Conv2d(in_channels=64, out_channels=oup_dim, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def _make_layer(self, block, in_planes, blocks):
        layers = []
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, in_planes * block.expansion * 2, kernel_size=1, stride=1, padding=0, bias=False),
                                   torch.nn.BatchNorm2d(in_planes * block.expansion * 2), self.relu,
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
        self.inplanes = in_planes * block.expansion * 2
        layers.append(block(self.inplanes, downsample=downsample)) 
        for i in range(1, blocks):
            layers.append(block(self.inplanes, downsample=None)) # channels keep same
        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_kernels)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = num_kernels[i], 1, 0
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2,
                                   padding=padding, output_padding=output_padding, bias=self.deconv_with_bias)
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def _upsample_layer(self, num_filters, up_num=2):
        layers = []
        for i in range(up_num):
            if up_num == 2:
                scale_factor = 2
            else:
                scale_factor = 4
            
            planes = num_filters[i]
            layers.extend(
                [
                    Conv(self.inplanes, planes, kernel_size=1, stride=1, bn=True, relu=True),
                    Conv(planes, planes, kernel_size=1, stride=1, bn=True, relu=True),  
                    Conv(planes, planes, kernel_size=1, stride=1, bn=False, relu=False), 
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                ]
            )
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)

        # checkpoint = torch.load(pretrained)
        # if isinstance(checkpoint, OrderedDict):
        #     state_dict = checkpoint
        # elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #     state_dict_old = checkpoint['state_dict']
        #     state_dict = OrderedDict()
        #     # delete 'module.' because it is saved from DataParallel module
        #     for key in state_dict_old.keys():
        #         if key.startswith('module.'):
        #             # state_dict[key[7:]] = state_dict[key]
        #             # state_dict.pop(key)
        #             state_dict[key[7:]] = state_dict_old[key]
        #         else:
        #             state_dict[key] = state_dict_old[key]
        # else:
        #     raise RuntimeError(
        #         'No state_dict found in checkpoint file {}'.format(pretrained))
        # self.load_state_dict(state_dict, strict=False)


    def forward(self, imgs):
        x = imgs #.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        c5 = self.deconv_layers(c4)
        out = self.final_layer(c5)
        #out = out.unsqueeze(1)
        return out#, None

    def calc_loss(self, nstack, preds, keypoints, heatmaps, masks):  # , ds_names):
        dets = preds[:, :, :self.num_parts]  # (batchsize, nstack, 19, oup_res, oup_res)
        tags = preds[:, :, self.num_parts:]

        keypoints = keypoints.cpu().long()
        batchsize = tags.size()[0]

        tag_loss = []
        for i in range(nstack):
            tag = tags[:, i].contiguous().view(batchsize, -1, 1)
            tag_loss.append(tagLoss(tag, keypoints))
        tag_loss = torch.stack(tag_loss, dim=1).cuda(tags.get_device())

        detection_loss = []
        for i in range(nstack):
            detection_loss.append(self.heatmapLoss(dets[:, i], heatmaps, masks))  # , ds_names))
        detection_loss = torch.stack(detection_loss, dim=1)

        losses = {
            'push_loss': tag_loss[:, :, 0],
            'pull_loss': tag_loss[:, :, 1],
            'detection_loss': detection_loss
        }

        return losses

def get_pose_net(cfg, args, is_train, **kwargs):
    model = PoseNet()
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        try:
            model._init_weights()
        except:
            pass
    return model
