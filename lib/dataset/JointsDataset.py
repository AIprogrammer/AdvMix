# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2, os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from .advaug import VanillaCombine, MixCombine
from imagecorruptions import corrupt, get_corruption_names


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, args, root, image_set, is_train, transform=None):

        self.args = args
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB
        # val mask/fg/parsing
        # self.val_fg = cfg.DATASET.VAL_FG
        # self.val_mask = cfg.DATASET.VAL_MASK
        # self.val_parsing = cfg.DATASET.VAL_PARSING

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform

        # self.get_varaug = VanillaCombine()
        self.get_varaug = MixCombine()
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError
    
    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5
        return center, scale

    def __len__(self,):
        return len(self.db)

    
    def __getitem__(self, idx):
        if self.args.sample_times == 1 or not self.is_train:
            input, target, target_weight, meta = self.get_clean(idx)
            return input, target, target_weight, meta
        else:
            input_list, target_list, target_weight_list, meta_list = [],[],[],[]
            meta, input_base = self.get_base(idx)
            get_cleans = ['clean', 'autoaug', 'gridmask']
            for sample in range(len(get_cleans)):
                get_clean = get_cleans[sample]
                input, target, target_weight, meta = self.get_var(meta, input_base, get_clean=get_clean)
                input_list.append(input)
                target_list.append(target)
                target_weight_list.append(target_weight)
                meta_list.append(meta)

            return input_list, target_list, target_weight_list, meta_list
    
    def get_base(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0
        
        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor  # 0.35
            rf = self.rotation_factor  # 45
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
        
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
        
        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        if 'style' in filename:
            datasetname = 'style'
        else:
            datasetname = 'clean'
            # logger.info(datasetname)

        if 'instance_index' not in db_rec:
            db_rec['instance_index'] = -1

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'dataset': datasetname,
            'instance_index':db_rec['instance_index']
        }

        return meta, input
    
    def get_var(self, meta, input, get_clean=False):
        joints = meta['joints']
        joints_vis = meta['joints_vis']
        
        inputs = {
            'data_numpy':input,
            'img_label':1,
            'joints_3d':joints,
            'joints_3d_vis':joints_vis,
            'dataset':meta['dataset']
        }

        inputs, _ = self.get_varaug((inputs, get_clean, self.args), self.transform)

        input = inputs['data_numpy']
        joints = inputs['joints_3d']
        joints_vis = inputs['joints_3d_vis']

        target, target_weight = self.generate_target(joints, joints_vis)

        if isinstance(target, list):
            for index in range(len(target)):
                target[index] = torch.from_numpy(target[index])
        else:
            target = torch.from_numpy(target)
        
        target_weight = torch.from_numpy(target_weight)

        if isinstance(target, list):
            return input, target[0], target_weight, meta
        else:
            return input, target, target_weight, meta

    def get_clean(self, idx):
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        ]
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            # IMREAD_COLOR : ALPHA= 
            # IMREAD_IGNORE_ORIENTATION : 
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            ### opencv read BGR index
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if self.args.random_corruption:
            # random corruptions
            print('random augmentation')
            data_numpy = corrupt(data_numpy, corruption_name=random.choice(corruptions), severity=random.randint(1,5))
        
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor  # 0.35
            rf = self.rotation_factor  # 45
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
        
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1


        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        
        target, target_weight = self.generate_target(joints, joints_vis)

        if isinstance(target, list):
            for index in range(len(target)):
                target[index] = torch.from_numpy(target[index])
        else:
            target = torch.from_numpy(target)
        
        target_weight = torch.from_numpy(target_weight)

        if 'instance_index' not in db_rec:
            db_rec['instance_index'] = -1

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'instance_index':db_rec['instance_index']
        }

        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            ### mean jointx, joiny
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            ### 16 points / all 0.2 variance 
            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16

            ### if joint center and bounding box center differ a lot, then deprecate
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def get_forground_image(self, img):
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (0, 0, img.shape[1] - 1, img.shape[0] - 1)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        cv2.imwrite(os.path.dirname((img)) + '_mask' + '/' + os.path.basename(img), img)
        return img

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32) # 17,1
        target_weight[:, 0] = joints_vis[:, 0] # 1st column

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = [np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32), np.zeros((self.num_joints, 2), dtype=np.float32)]
            # only in range tmp size == gaussian
            tmp_size = self.sigma * 3
            if False:
                tmp_hm = self.heatmap_size
                for joint_id in range(self.num_joints):
                    heatmap_vis = joints_vis[joint_id, 0]
                    target_weight[joint_id] = heatmap_vis
                    feat_stride = self.image_size / tmp_hm
                    mu_x = joints[joint_id][0] / feat_stride[0]
                    mu_y = joints[joint_id][1] / feat_stride[1]
                    # Check that any part of the gaussian is in-bounds
                    ul = [mu_x - tmp_size, mu_y - tmp_size]
                    br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                    if ul[0] >= tmp_hm[0] or ul[1] >= tmp_hm[1] or br[0] < 0 or br[1] < 0:
                            target_weight[joint_id] = 0
                                    
                    if target_weight[joint_id] == 0:
                        continue
                    
                    x = np.arange(0, tmp_hm[0], 1, np.float32)
                    y = np.arange(0, tmp_hm[1], 1, np.float32)
                    y = y[:, np.newaxis]

                    v = target_weight[joint_id]
                    if v > 0.5:
                        target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

            else:
                for joint_id in range(self.num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                    mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                    # Check that any part of the gaussian is in-bounds, otherwise kpt not in image size wrong
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    
                    if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        target_weight[joint_id] = 0
                        continue

                    # Generate gaussian
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized, we want the center value to equal 1
                    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    v = target_weight[joint_id]
                    if v > 0.5:
                        target[0][joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                    
                        target[1][joint_id] = np.array([mu_x, mu_y], dtype=np.float32)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight





