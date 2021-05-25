import torch
import torch.nn as nn
import numpy as np
import models

def gc_slim(model, cfg, args, ):
    #****************model sliming********************
    # layers = args.layers

    # args.layers < len(nn.BN)
    all_layer = 0
    for k, m in enumerate((model.modules())):
        if isinstance(m, nn.Conv2d):
            all_layer += 1
    
    layers = all_layer
    print('pruning the num of layers is ', layers)

    total = 0
    i = 0
    # baseline prune don't have bn layer
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                total += m.weight.data.shape[0]

    bn = torch.zeros(total)  # by channel
    index = 0
    i = 0
    # 有的层没有bn怎么办
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                # print('bn weight shape', m.weight.data.shape)  # [channel]
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size
    
    y, j = torch.sort(bn)
    thre_index = int(total * args.percent)  # only select topk
    
    if thre_index == total:
        thre_index = total - 1
    
    thre_0 = y[thre_index]  # 

    #****************获取针对分组卷积剪枝每层的基数（base_bumber）********************
    nums = []
    channels = []
    groups = []
    prune_base_num = []
    j = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            print('conv2d weight shape', m.weight.data.shape) # [out_channel, in_channel, 1, 1]
            s_0 = m.weight.data.data.size()[0]
            s_1 = m.weight.data.data.size()[1]
            nums.append(s_0)
            channels.append(s_1)
    
    print('nums and channels is ', nums) # Conv + 1 * dw
    print(channels)
    print('the lenght of channels is ', len(channels))

    r"""
    processing group conv [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 48, 1, 1, 1, 0]
    prune base num is     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 48, 48, 1, 1]
    """
    
    # conv总数； 为什么要这样做除法; 一般这样计算group分组数目吗  [out_channel, in_channel, k, k]
    # 因为正常的卷积的话上一层的输出channel除以下一层的输出channel应该等于，否则就是等于group的数目，这里需要特殊对待

    while  j < len(nums) - 1:
        groups.append(int(nums[j] / channels[j+1])) # 包含没有BN层的Conv
        j += 1
    
    print('processing group conv', groups)
    j = 0

    # 为什么这样做，不过如果是零的话正好去除
    while  j < len(groups) - 1:
        for i in range(1, (groups[j] * groups[j+1])+1):
            if i % groups[j] == 0 and i % groups[j+1] == 0:
                prune_base_num.append(i)
                break
        j += 1

    print('prune base num is', prune_base_num)

    #********************************预剪枝*********************************
    pruned = 0
    slim_cfg_0 = {}
    slim_cfg = {}
    slim_cfg_mask = {}
    i = 0

    for k, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1

                # 如果BN层的weight 与 上一层的conv weight[1]不相同，那么就是dw卷积
                weight_copy = m.weight.data.clone()
                mask = weight_copy.abs().gt(thre_0).float()
                # BN有些不一样，必须特殊处理
                remain_channels = torch.sum(mask)  # 当如果是group卷积时候，mask的维度会发生变化 in_channel 一定是1/ in_ch/ groups 固定数值， 不会发生变化， 选择的是out_channel

                if remain_channels == 0:
                    print('\r\n!please turn down the prune_ratio!\r\n')
                    remain_channels = 1
                    mask[int(torch.argmax(weight_copy))]=1

                # ******************分组卷积剪枝******************
                # base_number = prune_base_num[i-1]
                base_number = 1
                v = 0
                n = 1
                #  base channel
                if remain_channels % base_number != 0:
                    if remain_channels > base_number:
                        while v < remain_channels:
                            n += 1
                            v = base_number * n
                        if remain_channels - (v - base_number) < v - remain_channels:
                            remain_channels = v - base_number
                        else:
                            remain_channels = v
                        if remain_channels > m.weight.data.size()[0]:
                            remain_channels = m.weight.data.size()[0]
                        remain_channels = torch.tensor(remain_channels)
                            
                        y, j = torch.sort(weight_copy.abs())
                        # 可以在这儿改变mask
                        thre_1 = y[-remain_channels]
                        mask = weight_copy.abs().ge(thre_1).float()
                
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                slim_cfg_0[name] = mask.shape[0]
                slim_cfg[name] = int(remain_channels)
                slim_cfg_mask[name] = mask.clone()
                print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                    format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))
    
    pruned_ratio = float(pruned/total)
    print('\r\n!预剪枝完成!')
    print('total_pruned_ratio: ', pruned_ratio)
    print('新模型参数 : ', slim_cfg)  

    # [22, 32, 30, 34, 33, 22, 23, 66, 64, 64, 65, 66, 43, 40, 45, 63] 0.1
    # 

    #********************************剪枝*********************************
    newmodel = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, args, is_train=False, slim_cfg=slim_cfg
    )

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)  # input_channel = 3
    end_mask = slim_cfg_mask[layer_id_in_cfg]
    i = 0

    print(model)
    print('='*20)
    print(newmodel)

    for [[name0, m0], [name1, m1]] in zip(model.name_modules(), newmodel.named_modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(slim_cfg_mask):  
                    end_mask = slim_cfg_mask[layer_id_in_cfg]
            # for the last layer BN
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
        
        elif isinstance(m0, nn.Conv2d):
            if i < layers - 1:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                print('==> debuging', idx0.shape, idx1.shape, m0.weight.data.shape, m1.weight.data.shape)  # [17, 72, 1, 1]) torch.Size([17, 18, 1, 1]
                try:
                    w = m0.weight.data[:, idx0, :, :].clone()
                except:
                    w = m0.weight.data.clone()
                
                m1.weight.data = w[idx1, :, :, :].clone() # out, in

                # group 卷积中设置bias=False / 以及当没有BN时候如何计算

                try:
                    m1.bias.data = m0.bias.data[idx1].clone()
                except:
                    pass
            
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
                m1.bias.data = m0.bias.data.clone()
        
        elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
    
    #******************************剪枝后model测试*********************************
    print('新模型: ', newmodel)
    print('新模型参数：', slim_cfg)
    print('**********剪枝后新模型测试*********')

    return newmodel, slim_cfg

def normal_slim(model, cfg, args):

    all_layer = 0
    for k, m in enumerate((model.modules())):
        if isinstance(m, nn.Conv2d):
            all_layer += 1
    
    layers = all_layer

    print('pruning the num of layers is ', layers)
    base_number = 1
    total = 0
    i = 0
    for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if i < layers - 1:
                    i += 1
                    total += m.weight.data.shape[0]

    # 确定剪枝的全局阈值 全局阈值，根据BN的gamma数值大小
    bn = torch.zeros(total)
    index = 0
    i = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size
    
    y, j = torch.sort(bn)
    thre_index = int(total * args.percent)
    if thre_index == total:
        thre_index = total - 1
    thre_0 = y[thre_index]

    #********************************预剪枝*********************************
    pruned = 0
    slim_cfg_0 = []
    slim_cfg = []
    slim_cfg_mask = []
    i = 0

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1

                weight_copy = m.weight.data.clone()
                mask = weight_copy.abs().gt(thre_0).float()
                remain_channels = torch.sum(mask)

                if remain_channels == 0:
                    print('\r\n!please turn down the prune_ratio!\r\n')
                    remain_channels = 1
                    mask[int(torch.argmax(weight_copy))]=1

                # ******************规整剪枝******************
                v = 0
                n = 1
                # remain channels 是 base_number的整数倍
                if remain_channels % base_number != 0:
                    if remain_channels > base_number:
                        while v < remain_channels:
                            n += 1
                            v = base_number * n
                        if remain_channels - (v - base_number) < v - remain_channels:
                            remain_channels = v - base_number
                        else:
                            remain_channels = v
                        if remain_channels > m.weight.data.size()[0]:
                            remain_channels = m.weight.data.size()[0]
                        remain_channels = torch.tensor(remain_channels)
                            
                        y, j = torch.sort(weight_copy.abs())
                        thre_1 = y[-remain_channels]
                        mask = weight_copy.abs().ge(thre_1).float()
                
                pruned = pruned + mask.shape[0] - torch.sum(mask) # 去除的channel总数目
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                slim_cfg_0.append(mask.shape[0])  # 原始channel数目
                slim_cfg.append(int(remain_channels)) # 剪枝之后所有的保留的channel数目
                slim_cfg_mask.append(mask.clone())# mask
                print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                    format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))
    
    pruned_ratio = float(pruned/total)
    print('\r\n!预剪枝完成!')
    print('total_pruned_ratio: ', pruned_ratio)

    #********************************剪枝*********************************
    newmodel = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, args, is_train=False, slim_cfg=slim_cfg
    )

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)  # input_channel = 3
    end_mask = slim_cfg_mask[layer_id_in_cfg]
    i = 0

    print(model)
    print('='*20)
    print(newmodel)

    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(slim_cfg_mask):  
                    end_mask = slim_cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
        
        elif isinstance(m0, nn.Conv2d):
            if i < layers - 1:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                print('==> debuging', m0.weight.data.shape, m1.weight.data.shape)  # torch.Size([48, 1, 3, 3]) torch.Size([45, 1, 3, 3])
                w = m0.weight.data[:, idx0, :, :].clone()
                m1.weight.data = w[idx1, :, :, :].clone()  # out, in
                m1.bias.data = m0.bias.data[idx1].clone()
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
                m1.bias.data = m0.bias.data.clone()
        
        elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
    
    #******************************剪枝后model测试*********************************
    print('新模型: ', newmodel)
    print('新模型参数：', slim_cfg)
    print('**********剪枝后新模型测试*********')

    return newmodel, slim_cfg

