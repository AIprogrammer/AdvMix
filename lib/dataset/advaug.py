from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import torch
import logging

logger = logging.getLogger(__name__)


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    
    def __repr__(self):
        return "AutoAugment ImageNet Policy"



class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img



def grid_aug(cfg, img, joints, joints_vis, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1., db_rec=None):
    if np.random.rand() > prob:
        return img, joints, joints_vis,db_rec
    db_rec['img_label'] = 1
    h = img.size(1)
    w = img.size(2)
    d1 = 2
    d2 = min(h, w)
    hh = int(1.5*h)
    ww = int(1.5*w)
    d = np.random.randint(d1, d2)
    if ratio == 1:
        l = np.random.randint(1, d)
    else:
        l = min(max(int(d*ratio+0.5),1),d-1)
    mask = np.ones((hh, ww), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    if use_h:
        for i in range(hh//d):
            s = d*i + st_h
            t = min(s+l, hh)
            mask[s:t,:] *= 0
    if use_w:
        for i in range(ww//d):
            s = d*i + st_w
            t = min(s+l, ww)
            mask[:,s:t] *= 0
    
    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

    mask = torch.from_numpy(mask).float()
    if mode == 1:
        mask = 1-mask

    tmp_mask = mask
    mask = mask.expand_as(img)
    if offset:
        offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float()
        offset = (1 - mask) * offset
        img = img * mask + offset
    else:
        img = img * mask
    
    for joint_id in range(cfg.joints_num):
        joint = joints[joint_id][:2]
        tmp_x = min(int(joint[0]), tmp_mask.shape[1] - 1)
        tmp_x = max(tmp_x, 0)
        tmp_y = min(int(joint[1]), tmp_mask.shape[0] - 1)
        tmp_y = max(tmp_y, 0)

        if tmp_mask[tmp_y, tmp_x] == 0:
            joints_vis[joint_id][0] = 0
            joints_vis[joint_id][1] = 0

    return img, joints, joints_vis,db_rec


class MixCombine(object):
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.autoaug = ImageNetPolicy(fillcolor=(128, 128, 128))
    def __call__(self, inputs, transform):
        db_rec, get_clean, cfg = inputs
        if get_clean != 'clean':
            if get_clean == 'autoaug':
                if db_rec['dataset'] != 'style' or not cfg.sp_style:
                    db_rec['img_label'] = 1
                    data_numpy = db_rec['data_numpy']
                    tmp_img = Image.fromarray(data_numpy.astype(np.uint8))
                    data_numpy = self.autoaug(tmp_img)
                    db_rec['data_numpy'] = np.array(data_numpy)
                db_rec['data_numpy'] = transform(db_rec['data_numpy'])
            
            elif get_clean == 'gridmask':
                db_rec['data_numpy'] = transform(db_rec['data_numpy'])
                if db_rec['dataset'] != 'style' or not cfg.sp_style:
                    rotate = 1
                    offset=False 
                    ratio = 0.5
                    mode=1
                    prob = 0.7
                    self.st_prob = prob
                    data_numpy = db_rec['data_numpy']
                    joints = db_rec['joints_3d']
                    joints_vis = db_rec['joints_3d_vis']

                    data_numpy, joints, joints_vis, db_rec = grid_aug(cfg, data_numpy, joints, joints_vis, True, True, rotate, offset, ratio, mode, prob, db_rec)
                    
                    db_rec['data_numpy'] = data_numpy
                    db_rec['joints_3d'] = joints
                    db_rec['joints_3d_vis'] = joints_vis
        
        else:
            db_rec['data_numpy'] = transform(db_rec['data_numpy'])

        return db_rec, cfg

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch
