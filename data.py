import os
import glob
import cv2

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class ImageNetDataset(Dataset):
    def __init__(self, dirpath, split, transform):
        super().__init__()
        self.dirpath = dirpath
        self.class_dict = dict()
        self.split = split
        self.transform = transform
        
        # class dict
        mapping_path = os.path.join(self.dirpath, "LOC_synset_mapping.txt")
        with open(mapping_path, "r") as f:
            lines = f.read().strip().split("\n")
        
        for class_num, line in enumerate(lines):
            ind = line.find(" ")
            k = line[:ind]
            v = line[ind:].strip().split(",")[0]
            self.class_dict[k] = [class_num, v]

        # image list
        if self.split == "train":
            self.img_lst = glob.glob(os.path.join(dirpath, "ILSVRC/Data/CLS-LOC/train/*/*"))
        elif self.split == "val":
            self.img_lst = glob.glob(os.path.join(dirpath, "ILSVRC/Data/CLS-LOC/val/*"))
        elif self.split == "test":
            self.img_lst = glob.glob(os.path.join(dirpath, "ILSVRC/Data/CLS-LOC/test/*"))
        else:
            raise ValueError("Check split")

    def __len__(self):
        return len(self.img_lst)
    
    def __getitem__(self, index):
        img_path = self.img_lst[index]
        label = torch.tensor(self.class_dict[os.path.basename(img_path).split("_")[0]][0], dtype=torch.long)
        img = read_image(img_path, ImageReadMode.RGB)
        img = self.transforms(img)
        label = nn.functional.one_hot(label, 1000)
        label = label.float()
        return img, label



class COCO(Dataset):
    def __init__(self, dirpath, split, size, transform):
        super().__init__()
        self.dirpath = dirpath
        self.split = split
        self.size = size
        self.transform = transform

        anno_path = os.path.join(dirpath, 'annotations/small_train_anno.txt')
        with open(anno_path, 'r') as f:
            self.data = f.read().strip().split('\n')
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, x1, y1, x2, y2, cat_id = self.data[index].split(' ')
        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))
        img_path = os.path.join(self.dirpath, img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        new_size = max(h, w)
        new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        if h > w:
            s = (h-w)//2
            new_img[:,s:s+w] = img
            x1 += s
            x2 += s
        else:
            s = (w-h)//2
            new_img[s:s+h,:] = img
            y1 += s
            y2 += s

        onehot_cat_id = nn.functional.one_hot(torch.tensor(int(cat_id), dtype=torch.long), 90).float()
        x1 = x1 * (self.size/new_size)
        x2 = x2 * (self.size/new_size)
        y1 = y1 * (self.size/new_size)
        y2 = y2 * (self.size/new_size)
        img = self.transform(new_img)
        return img, torch.tensor([x1, y1, x2, y2]), onehot_cat_id
    
if __name__  == "__main__":
    traindata = COCO("/data/coco", split='train', transform=None)
    