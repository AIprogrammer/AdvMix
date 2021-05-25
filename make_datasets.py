from PIL import Image
import numpy as np
import time, os
from glob import glob
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import Dataset
from imagecorruptions import corrupt, get_corruption_names


parser = argparse.ArgumentParser(description='Apply different corruption types to official validation dataset, e.g., COCO, MPII, OCHuman, etc.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root_dir', type=str, default="./data", help='Root directory of data.')
parser.add_argument('--data_dir', type=str, help='Directroy of images.')
parser.add_argument('--dataset', type=str, default="COCO", help='Dataset to process.')
args = parser.parse_args()

class make_data(Dataset):
    def __init__(self, root_dir, data_dir, dataset):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.dataset = dataset
        self.imglist = glob(self.root_dir + '/' + self.data_dir + '/*.jpg')

    def __getitem__(self, index):
        img = self.imglist[index]
        self.process(img)        
        return 0

    def __len__(self,):
        return len(self.imglist)
    
    def process(self,img):
        image = np.asarray(Image.open(img))
        for corruption in get_corruption_names('all'):
            for severity in range(5):
                corrupted = corrupt(image, corruption_name=corruption, severity=severity+1)
                corrupted_path = os.path.join(self.root_dir, self.dataset + '-C', corruption, str(severity), os.path.basename(img))
                if not os.path.exists(os.path.dirname(corrupted_path)):
                    os.makedirs(os.path.dirname(corrupted_path))
                Image.fromarray(corrupted).save(corrupted_path)

if __name__ == '__main__':
    root_dir = args.root_dir
    data_dir = args.data_dir
    which_dataset = args.dataset
    d_dataset = make_data(root_dir, data_dir, which_dataset)
    print("To process {} images. ".format(len(d_dataset)))
    distorted_dataset_loader = torch.utils.data.DataLoader(
            d_dataset, batch_size=64, shuffle=False, num_workers=8)

    for _ in tqdm(distorted_dataset_loader): continue
