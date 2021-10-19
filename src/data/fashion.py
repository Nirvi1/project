# -*- coding:utf-8 -*-
"""
File: __init__.py
File Created: 2021-09-23
Author: Nirvi Badyal
"""
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


FashionMEAN = [0.485, 0.456, 0.406]
FashionSTD = [0.229, 0.224, 0.225]
PATH_TO_DATA = "/ml_model/second/DeepFashion/data"


class FashionDataset(Dataset):
    CLASSES = None
    def __init__(self, img_size, transform, data_path,
                 img_file, attr_file,
                 bbox_file, landmark_file):
        self.img_size = img_size
        self.data_path = data_path
        self.transform = transform

        self.img_lst = []
        with open(os.path.join(data_path, img_file), 'r') as fr:
            for x in tqdm(fr):
                xpath = os.path.join(self.data_path, x.strip())
                img = Image.open(xpath).convert('RGB')
                self.img_lst.append(img)
        
        if attr_file is not None:
            self.attr = np.loadtxt(os.path.join(data_path, attr_file), dtype=np.int64)
        else:
            self.attr = None
        
        self.bboxes = np.loadtxt(os.path.join(data_path, bbox_file), usecols=(0, 1, 2, 3))
        self.landmarks = np.loadtxt(os.path.join(data_path, landmark_file))
        
    @staticmethod
    def bbox_fix(bbox_cor, bpad=10):     
        bbox_w, bbox_h = float(int(bbox_cor[2]) - int(bbox_cor[0])), int(bbox_cor[3]) - int(bbox_cor[1])
        x1, x2 = max(0, int(bbox_cor[0]) - bpad), int(bbox_cor[2]) + bpad
        y1, y2 = max(0, int(bbox_cor[1]) - bpad), int(bbox_cor[3]) + bpad
        return x1, x2, y1, y2
        
    def __getitem__(self, idx):
        img = self.img_lst[idx]
        # print(img.size)
        
        # bbox
        x1, x2, y1, y2 = self.bbox_fix(self.bboxes[idx])
        bbox_w, bbox_h = x2 - x1, y2 - y1
        img = img.crop(box=(x1, y1, x2, y2))
        
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = self.transform(img)
#         print(img.size())
        
        # landmark
        origin_landmark = self.landmarks[idx]
        landmark = []
        for i, l in enumerate(origin_landmark):
            if i % 2 == 0:  # x
                l_x = max(0, l - x1)
                l_x = float(l_x) / bbox_w * self.img_size[0]
                landmark.append(l_x)
            else:  # y
                l_y = max(0, l - y1)
                l_y = float(l_y) / bbox_h * self.img_size[1]
                landmark.append(l_y)
        landmark = torch.from_numpy(np.array(landmark)).float()

        # label
        if self.attr is not None:
            attr = torch.from_numpy(self.attr[idx])
            # attr = [F.one_hot(attr[i], num_classes=ATTR_OUT[i]) for i in range(6)]
        else:
            attr =torch.zeros((6,))
            # attr = [torch.zeros((ATTR_OUT[i],)) for i in range(6)]
        
        data = {'img': img, 'attr': attr, 'landmark': landmark}
        return data
    
    def __len__(self):
        return len(self.img_lst)


def get_data_tra(data_path=PATH_TO_DATA, batch_size=64, img_size=[224, 224]):
    trans_tra = transforms.Compose(
        [ transforms.Resize(img_size),
         #transforms.RandomCrop((112,112)), 
         #transforms.Resize(img_size),
         
         transforms.RandomHorizontalFlip() ,
         #transforms.RandomCrop((128,128)),
         transforms.RandomRotation(degrees=(0,10)),
         #transforms.ColorJitter(
         #  brightness=0.1, contrast=0.1),
         #transforms.Grayscale(num_output_channels=3) ,
         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)) ,
         transforms.ToTensor(),
         #transforms.Normalize(
         #  mean=FashionMEAN, std=FashionSTD)
         ])
         #transforms.Normalize(
         # mean=FashionMEAN, std=FashionSTD)])
    ds_tra = FashionDataset(img_size=img_size, transform=trans_tra, 
        data_path=data_path,
        img_file="split/train.txt", 
        attr_file="split/train_attr.txt",
        bbox_file="split/train_bbox.txt", 
        landmark_file="split/train_landmards.txt")
    dl_tra = DataLoader(ds_tra, batch_size=batch_size, shuffle=True, num_workers=8)
    return dl_tra


def get_data_val(data_path=PATH_TO_DATA, batch_size=50, img_size=[224, 224]):
    trans_val = transforms.Compose(
        [
         transforms.Resize(img_size),
         #transforms.RandomHorizontalFlip(),
         #transforms.ColorJitter(
         #     brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
         transforms.ToTensor(),
         #transforms.Normalize(
         #   mean=FashionMEAN, std=FashionSTD)
       ])
    ds_val = FashionDataset(img_size=img_size, transform=trans_val, 
        data_path=data_path,
        img_file="split/val.txt", 
        attr_file="split/val_attr.txt",
        bbox_file="split/val_bbox.txt", 
        landmark_file="split/val_landmards.txt")
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=8)
    return dl_val


def get_data_tes(data_path=PATH_TO_DATA, batch_size=50, img_size=[224, 224]):
    trans_tes = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor()])
         #transforms.Normalize(
         #   mean=FashionMEAN,std=FashionSTD)])
    ds_tes = FashionDataset(img_size=img_size, transform=trans_tes, 
        data_path=data_path,
        img_file="split/test.txt", 
        attr_file=None,
        bbox_file="split/test_bbox.txt", 
        landmark_file="split/test_landmards.txt")
    dl_tes = DataLoader(ds_tes, batch_size=batch_size, shuffle=False)
    return dl_tes
