import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils

def ProductNet(arch, pretrained=None):
    print("=> creating model '{}'".format(arch))
    '''
    if pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
        print("=> using pre-trained parameters '{}'".format(pretrained))
        model = pretrainedmodels.__dict__[arch](num_classes=1000,
                                                     pretrained=pretrained)                                                                                                                                 
    else:
        model = pretrainedmodels.__dict__[arch]()
    '''
    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=pretrained)
    # layer0--layer4 avg_pool last_linear
    senet = model.state_dict()
    if arch == 'se_resnet50':
        resume = 'pre_model_dict/se_resnet50-ce0d4300.pth'
    if arch == 'se_resnet101':
        resume = 'pre_model_dict/se_resnet101-7e38fcc6.pth'
    if arch == 'se_resnet152':
        resume = 'pre_model_dict/se_resnet152-d17c99b7.pth'
    if arch == 'senet154':
        resume = 'pre_model_dict/senet154-c7b49a05.pth'
    if arch == 'inceptionresnetv2':
        resume = 'pre_model_dict/inceptionresnetv2-520b38e4.pth'
    if arch =='nasnetalarge':
        resume = 'pre_model_dict/nasnetalarge-a1897284.pth'
    if arch == 'se_resnext101_32x4d':
        resume = 'pre_model_dict/se_resnext101_32x4d-3b2fe3d8.pth'
    if arch == 'dpn131':
        resume = 'pre_model_dict/dpn131-7af84be88.pth'
    if arch == 'dpn107':
        resume = 'pre_model_dict/dpn107_extra-b7f9f4cc9.pth'
    if arch == 'inceptionv4':
        resume = 'pre_model_dict/inceptionv4-8e4777a0.pth'

    pretrained_dict = torch.load(resume)
    model_dict = {k:v for k, v in pretrained_dict.items() if (k in senet) and k.split('.')[0] != 'last_linear'}    
    senet.update(model_dict)
    model.load_state_dict(senet)
    if arch == 'inceptionresnetv2':
        model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
    elif arch != 'dpn107' and arch != 'dpn131':
        model.avg_pool = nn.AdaptiveAvgPool2d(1)# 1*1*in_features
        model.last_linear = nn.Linear(model.last_linear.in_features, 251)# batch_size, num_cls
    else:
        model.last_linear = nn.Conv2d(model.last_linear.in_channels, 251, kernel_size=(1,1), stride=1)
    # model.last_linear = nn.Linear(model.last_linear.in_features, 637)
    # model.relu = nn.ReLU(inplace=True)
    
    return model

class Slice(nn.Module):
    def __init__(self, num_channel, ks1, ks2):
        super(Slice, self).__init__()
        self.slice = nn.Sequential(
            nn.Conv2d(3, num_channel, ks1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.MaxPool2d(ks2),
        )
        # self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.slice(x)
        # x2 = self.pool(x)
        # x = torch.cat((x1, x2), 1)
        return x

class iFood(nn.Module):
    def __init__(self, arch, num_classes, pretrained):
        super(iFood, self).__init__()
        model = pretrainedmodels.__dict__[arch](1000, pretrained=pretrained)
        model.last_linear = nn.Linear(2048, num_classes)
        # self.model = nn.Sequential(*list(model.children())[:-2])
        if arch == 'se_resnet50':
            net = 'pre_model_dict/se_resnet50-ce0d4300.pth'
        if arch == 'se_resnet101':
            net = 'pre_model_dict/se_resnet101-7e38fcc6.pth'
        if arch == 'se_resnet152':
            net = 'pre_model_dict/se_resnet152-d17c99b7.pth'
        if arch == 'senet154':
            # net = 'pre_model_dict/senet154-c7b49a05.pth'
            net = "/export/home/wangjiahang/FGVC/DCL_ifood/zhd/senet154_secondmodel/checkpoint/checkpoint_epoch_027_prec3_90.812_pth.tar"
        if arch =='se_resnext101_32x4d':
            net = 'pre_model_dict/se_resnext101_32x4d-3b2fe3d8.pth'
        if arch == 'dpn131':
            net = 'pre_model_dict/dpn131-7af84be88.pth'
        if arch == 'dpn107':
            net = 'pre_model_dict/dpn107_extra-b7f9f4cc9.pth'
        print('tune from imagenet')
        print('base on ' + arch)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(net)
        # print(pretrained_dict.keys())
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (k.split('.')[0] != 'last_linear')}
        pretrained_dict = {k[7:]:v for k, v in pretrained_dict.items() if k.split('.')[1] != 'last_linear' and k[7:] in model.state_dict()}
        assert len(pretrained_dict) != 0
        model_dict.update(pretrained_dict) # model_dict
        model.load_state_dict(model_dict)
        
        for i in range(5):
            setattr(self, 'layer{}'.format(i), getattr(model, 'layer{}'.format(i)))
        self.slice = Slice(1024, (299, 5), (1, 295))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 251),
            nn.BatchNorm1d(251),
            nn.Dropout(0.5),
        )
        
    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)#2048
        x6 = self.slice(x)
        x7 = self.pool(x5)
        x = torch.cat((x6, x7), 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = ProductNet('dpn107', 'None')
    print(model.state_dict)
    print(model.relu)


    






