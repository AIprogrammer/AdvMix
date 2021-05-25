# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

mm_path = osp.join(this_dir, '..', 'lib/poseeval/py-motmetrics')
add_path(mm_path)

# mmdectection
lib_path = osp.join(this_dir, '..', 'mmdetection')

lib_path = osp.join(this_dir, '..', 'Synchronized_BatchNorm')
add_path(lib_path)

if __name__ == '__main__':
    import os.path as osp
    this_dir = osp.dirname(__file__)
    print(this_dir)
    ### only current path and sys append path