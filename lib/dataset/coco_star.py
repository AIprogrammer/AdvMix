import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from collections import OrderedDict
import logging
import os
import sys
import time, torch
import random

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np

from star.core.nms import oks_nms, soft_oks_nms
from .custom import CustomDataset
from ..utils.registry import DATASETS
from ..pipelines import Compose
from .utils import NullWriter
from pycocotools import mask


logger = logging.getLogger(__name__)


@DATASETS.register_module
class CocoDataset(CustomDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

        [6,7],[7,8],[9,10],[10,11],[12,13],[13,14],[15,16],[16,17]
    '''
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self, data_cfg, train_cfg, test_cfg, is_train=True, transform=None):
        cfg = data_cfg
        super().__init__(cfg, is_train, transform)
        self.cfg.rank = cfg.rank
        self.cfg.world_size = cfg.world_size
        self.cfg.nms_thre = test_cfg.NMS_THRE 
        self.cfg.image_thre = test_cfg.IMAGE_THRE 
        self.cfg.soft_nms = test_cfg.SOFT_NMS 
        self.cfg.oks_thre = test_cfg.OKS_THRE 
        self.cfg.in_vis_thre = test_cfg.IN_VIS_THRE 
        self.cfg.bbox_file = test_cfg.COCO_BBOX_FLIP 
        self.cfg.use_gt_bbox = test_cfg.USE_GT_BBOX 
        self.cfg.image_width = cfg.IMAGE_SIZE[0]
        self.cfg.image_height = cfg.IMAGE_SIZE[1]

        self.cfg.aspect_ratio = self.cfg.image_width * 1.0 / self.cfg.image_height

        self.cfg.sub_data_name = cfg.sub_data_name
        self.cfg.model_supervise_channel = cfg.model_supervise_channel
        self.cfg.model_select_channel = cfg.model_select_channel
        self.cfg.model_supervise_channel_onehot = cfg.model_supervise_channel_onehot
        self.cfg.model_select_channel_onehot = cfg.model_select_channel_onehot

        self.cfg.num_joints = cfg.num_keypoints
        self.cfg.num_heatmap = cfg.NUM_HEATMAP
        self.cfg.num_merge_joints = cfg.num_merge_keypoints


        self.cfg.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]

        self.cfg.parent_ids = None
        self.cfg.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.cfg.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.cfg.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.cfg.num_joints, 1))

        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite
        self.coco = COCO(self.cfg.annotations_path)
        sys.stdout = oldstdout
        
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )
        self.image_set_index = self.coco.getImgIds()
    
        self.num_images = len(self.image_set_index)

        self.db = self._get_db()

        if is_train and cfg.select_data:
            self.db = self.select_data(self.db)

        if self.cfg.mini_data != {}:
            if self.cfg.mini_data['coco'] != 1:
                self.db = random.sample(self.db, int(len(self.db) * self.cfg.mini_data['coco']))
        if is_train:
            if self.cfg.ratio != {}:
                if self.cfg.ratio['coco'] != 1:
                    self.db = self.db * int(self.cfg.ratio['coco'])

            if self.cfg.rank == 0:
                print ('=> num_images: {}'.format(self.num_images))
                print ('=> load {} samples'.format(len(self.db)))


    
    def _get_db(self):
        if self.cfg.is_train or self.cfg.use_gt_bbox:
            gt_db = self._load_coco_keypoint_annotations()
        else:
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        mask_path = self.mask_path_from_index(index)
        
        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue
            
            joints_3d = np.zeros((self.cfg.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.cfg.num_joints, 3), dtype=np.float)
            for ipt in range(self.cfg.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0
            center, scale = self._box2cs(obj['clean_bbox'][:4])
            instance_seg = obj['segmentation']
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'dataset': 'coco',
                'imgnum': 0,
                'mask_image':mask_path,
                'instance_seg':instance_seg
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        dice1 = random.random()
        if self.cfg.is_train and dice1 > 0.7:
            dice_x = random.random()
            center[0] = center[0] + (dice_x - 0.5) * w * 0.4
            dice_y = random.random()
            center[1] = center[1] + (dice_y - 0.5) * h * 0.4

        if w > self.cfg.aspect_ratio * h:
            h = w * 1.0 / self.cfg.aspect_ratio
        elif w < self.cfg.aspect_ratio * h:
            w = h * self.cfg.aspect_ratio
        scale = np.array(
            [w * 1.0 , h * 1.0],
            dtype=np.float32)

        # assert center[0]>-0.5, center
        
        scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        if random.random() < self.cfg.shuffleinmask_prob and self.cfg.is_train:
            image_path = os.path.join(
                self.cfg.image_path + '_shuffleinmask', '%012d.jpg' % index)
        else:
            image_path = os.path.join(
                self.cfg.image_path, '%012d.jpg' % index)

        return image_path

    def mask_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        if self.cfg.is_train:
            mask_path = os.path.join(
                self.cfg.image_path + "_mask", '%012d.png' % index)
        else:
            mask_path = os.path.join(
                'dataset/coco/val2017_mask', '%012d.png' % index)
        return mask_path


    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.cfg.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.cfg.bbox_file)
            return None
        if self.cfg.rank == 0:
            print ('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            
            img_name = self.image_path_from_index(det_res['image_id'])
            mask_path = self.mask_path_from_index(det_res['image_id'])

            box = det_res['bbox']
            score = det_res['score']

            if score < self.cfg.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.cfg.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.cfg.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'dataset': 'coco',
                'rotation': 0,
                'imgnum': 0,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'mask_image':mask_path,
                'instance_seg':None
            })
        if self.cfg.rank == 0:
            print ('=> Total boxes after fliter low score@{}: {}'.format(
                self.cfg.image_thre, num_boxes))
        return kpt_db


    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_results_{}.json'.format(rank)
        )

        # person x (keypoints)
        _kpts = []
        # print ('len(img_path), len(all_boxes), len(pr', len(img_path), len(all_boxes), len(preds), flush=True)
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4]),
                
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.cfg.num_joints
        in_vis_thre = self.cfg.in_vis_thre
        oks_thre = self.cfg.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.cfg.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file, res_folder)
        try:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        except:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file, res_folder):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        # print (f'rank {self.cfg.rank} ', end='', flush=True)

        if self.cfg.rank == 0:
            print ('\n => writing results json to %s' % res_file,flush=True)
            time.sleep(10)
            while_cnt = 1
            while(while_cnt):
                while_cnt += 1
                time.sleep(1)
                clean_json_list = []
                json_list = os.listdir(res_folder)
                for json_name in json_list:
                    if json_name[:18] == 'keypoints_results_':
                        clean_json_list.append(json_name)
                if len(clean_json_list) == self.cfg.world_size:
                    break
                if while_cnt % 10 == 9:
                    print (f'in evalution.. num pred {len(clean_json_list)} {self.cfg.world_size}')
            merge_result = []
            for file in clean_json_list:
                data = json.load(open(os.path.join(res_folder, file)))
                merge_result += data
                sh = f'rm {os.path.join(res_folder, file)}'
                os.system(sh)

            with open(res_file, 'w') as f:
                json.dump(merge_result, f, sort_keys=True, indent=4)

        try:
            json.load(open(res_file))
        except Exception:
            print ('json error...', flush=True)
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.cfg.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.cfg.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):

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
