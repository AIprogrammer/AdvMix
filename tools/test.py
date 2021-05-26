import os
import torch
import argparse

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import star
from star.config.config import parse_args, merge_cfg
from star.models import build_detector
from star.data import build_dataset
from star.utils import create_logger

from star.core.loss import build_loss
from star.core.function import save_checkpoint
from star.core.function import get_optimizer
from star.core.function import build_dataloader, build_trainer, build_tester
import shutil
from testmap import create_pifpaf
from collections import defaultdict
import pandas as pd


def do_python_keypoint_eval(res_file, res_folder):

    nullwrite = NullWriter()
    oldstdout = sys.stdout
    sys.stdout = nullwrite

    coco_dt = self.coco.loadRes(res_file)
    coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    sys.stdout = oldstdout


    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    return info_str


def main():
    args = parse_args()
    cfg = merge_cfg(args)

    logger, final_output_dir, tb_log_dir, writer_dict = create_logger(cfg, 'train')

    model, begin_epoch, best_perf, optimizer,_ = build_detector(cfg, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    criterion = build_loss(cfg)

    valid_annotations = cfg.data.data_cfg.valid_annotations
    valid_image_path = cfg.data.data_cfg.valid_image_path

    if not type(valid_annotations) == list:
        valid_annotations = [valid_annotations]
        valid_image_path = [valid_image_path]

    res = []
    for i in range(len(valid_annotations)):        
        if args.use_det_bbox:
            cfg.test_cfg.use_gt_bbox = False
            print('===>> Test using det bbox')
        
        cfg.data.data_cfg.valid_image_path = valid_image_path[i]
        _, _, valid_loader, _, valid_dataset = build_dataloader(cfg, is_train=False) # return both train dataloader and valid dataloader
        cfg.test_cfg.tester = True
        tester = build_tester(cfg)

        perf_indicator = tester(cfg, valid_loader, valid_dataset, model, criterion, -1,
                final_output_dir, tb_log_dir, writer_dict)
        res.append(perf_indicator)

        if args.test_robustness:
            print('====> test robustness ...', flush=True)
            distortions = [
                'gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                'snow', 'frost', 'fog', 'brightness',
                'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
            ]

            which_dataset = cfg.data.data_cfg.valid_image_path.split('/')[1]
            for distortion_name in distortions:
                ### 5 severity in each distortions
                for severity in range(5):
                    if which_dataset == 'coco':
                        base_path = 'dataset/corrupted_data/coco/images/val2017_ascoco/{}/{}'.format(distortion_name, severity)
                    elif which_dataset == 'ochuman':
                        base_path = 'dataset/corrupted_data/ochuman/images/{}/{}'.format(distortion_name, severity)
                        # base_path = 'dataset/ochuman/images'
                    elif which_dataset == 'mpii':
                        base_path = 'dataset/corrupted_data/mpii/images/{}/{}'.format(distortion_name, severity)
                       
                    cfg.data.data_cfg.valid_image_path = base_path
                    _, _, valid_loader, _, valid_dataset = build_dataloader(cfg, is_train=False)
                    cfg.test_cfg.tester = True
                    tester = build_tester(cfg)
                    sepa_output_dir = final_output_dir + '/' + distortion_name + '/' + str(severity)
                    print(sepa_output_dir)
                    perf_indicator = tester(cfg, valid_loader, valid_dataset, model, criterion, -1,
                            sepa_output_dir, tb_log_dir, writer_dict)
                    res.append(perf_indicator)

    if which_dataset == 'mpii':
        get_final_results_mpii(res, distortions, final_output_dir, args.exp_id, mode='td')
    else:
        mode = 'bu' if cfg.model.type == 'BottomUp' else 'td'
        get_final_results(res, distortions, final_output_dir, args.exp_id, mode=mode)

def get_final_results(mAP, distortions, final_output_dir, exp_id,mode='td'):
    # if mode == 'td':
    #     keyword = '| TopDown |'
    # elif mode == 'bu':
    #     keyword = '| BottomUp |'

    dic = {}
    # log_dir = final_output_dir + '/' + exp_id + '.log'
    # lines = [_.strip() for _ in open(log_dir, 'r').readlines()]
    # mAP = []
    # for line in lines:
    #     if keyword in line:
    #         mAP.append(float(line.split(keyword  + ' ')[1][:5]))
    
    assert len(mAP) == 96, 'Result length'
    
    dic['clean_mAP'] = [mAP.pop(0)]

    print(mAP)

    all_tmp = 0
    for dis in distortions:
        tmp = []
        for i in range(5):
            tmp.append(mAP[distortions.index(dis) * 5 + i])
        dic[dis] = [sum(tmp) / len(tmp)]
        if dis in distortions[:15]:
            all_tmp += dic[dis][0]
    
    dic['mean_corrupted_AP'] = [all_tmp / 15]
    dic['rAP'] = dic['mean_corrupted_AP'][0] / dic['clean_mAP'][0]

    dataframe = pd.DataFrame(dic)
    columns = ['clean_mAP', 'mean_corrupted_AP', 'rAP'] + distortions 
    dataframe.to_csv(final_output_dir + '/' + exp_id + ".csv", index=False,sep=',', columns=columns)


def get_final_results_mpii(mean, distortions, final_output_dir, exp_id,mode='td'):
    # if mode == 'td':
    #     keyword = '| TopDown |'
    # elif mode == 'bu':
    #     keyword = '| BottomUp |'

    dic = {}
    # log_dir = final_output_dir + '/' + exp_id + '.log'
    # lines = [_.strip() for _ in open(log_dir, 'r').readlines()]
    # mean = []
    # for line in lines:
    #     if keyword in line:
    #         mean.append(float(line.split('| ')[9].strip()))

    assert len(mean) == 96, 'Result length'
    
    dic['clean_mean'] = [round(mean.pop(0),3)]

    print(mean)

    all_tmp = 0
    for dis in distortions:
        tmp = []
        for i in range(5):
            tmp.append(mean[distortions.index(dis) * 5 + i])
        dic[dis] = [round(sum(tmp) / len(tmp),3)]
        if dis in distortions[:15]:
            all_tmp += dic[dis][0]
    
    dic['mean_corrupted_mean'] = [round(all_tmp / 15,3)]
    dic['rmean'] = round(dic['mean_corrupted_mean'][0] / dic['clean_mean'][0],3)

    dataframe = pd.DataFrame(dic)
    columns = ['clean_mean', 'mean_corrupted_mean', 'rmean'] + distortions 
    dataframe.to_csv(final_output_dir + '/' + exp_id + ".csv", index=False,sep=',', columns=columns)

if __name__ == '__main__':
    main()